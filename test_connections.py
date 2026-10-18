import json
import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import urllib.error

import connections
import quorum as q
import runs
import webapp


class ConnectionTests(unittest.TestCase):
    def setUp(self):
        runs_dir = tempfile.TemporaryDirectory()
        self.addCleanup(runs_dir.cleanup)
        patcher = patch.dict(os.environ, {"QUORUM_RUNS_DIR": runs_dir.name})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_openai_override_is_independent(self):
        with tempfile.TemporaryDirectory() as td:
            config = Path(td) / "config.toml"
            config.write_text('[[panel]]\nname="gpt"\nprovider="openai"\nmodel="gpt-6.1-sol"\nauth="cli"\n')
            connections.save_claude(config, "api", "claude-secret")
            connections.save_claude(config, "api", "openai-secret", provider="openai")
            with patch.object(q, "CONFIG_PATH", config):
                model = q.load_config().panel[0]
                self.assertEqual((model.auth, model.api_key, model.model), ("api", "openai-secret", "gpt-6.1-sol"))
            connections.save_claude(config, "cli", remove=True, provider="openai")
            self.assertEqual(connections.read(config)["anthropic"]["api_key"], "claude-secret")
            self.assertEqual(connections.read(config)["openai"]["api_key"], "")

    def test_openai_responses_tools_and_usage(self):
        model = q.Model("gpt", "openai", "gpt-6.1-sol", api_key="test-secret")
        response = {"status": "completed", "usage": {"input_tokens": 120, "output_tokens": 25},
                    "output": [{"type": "reasoning"}, {"type": "message", "content": [
                        {"type": "output_text", "text": "review"}]}]}
        with patch.object(q, "_http_json", return_value=response) as http:
            self.assertEqual(q.call_model(model, "question", True), "review")
            self.assertEqual(http.call_args.args[0], "https://api.openai.com/v1/responses")
            self.assertEqual(http.call_args.args[1]["tools"], [{"type": "web_search"}])
            self.assertEqual((model.tokens_in, model.tokens_out, model.tokens_est), (120, 25, False))

    def test_openai_key_check_does_not_save_on_test(self):
        class Request:
            headers = {"content-type": "application/json", "host": "quorum.test", "origin": "https://quorum.test"}

            async def json(self):
                return {"auth": "api", "api_key": "test-secret", "test_only": True}

        model = q.Model("gpt", "openai", "gpt-6.1-sol", auth="cli")
        with patch.object(q, "load_config", return_value=SimpleNamespace(panel=[model])), \
             patch.object(webapp.urllib.request, "urlopen") as upstream, \
             patch.object(connections, "save_claude") as save:
            upstream.return_value.__enter__.return_value.read.return_value = b'{}'
            response = asyncio.run(webapp.gpt_connection(Request()))
            self.assertEqual(response.status_code, 200)
            request = upstream.call_args.args[0]
            self.assertEqual(request.full_url, "https://api.openai.com/v1/models/gpt-6.1-sol")
            self.assertEqual(request.get_header("Authorization"), "Bearer test-secret")
            self.assertNotIn(b"test-secret", response.body)
            save.assert_not_called()

    def test_invalid_api_key_does_not_change_connection(self):
        class Request:
            headers = {"content-type": "application/json", "host": "quorum.test", "origin": "https://quorum.test"}

            async def json(self):
                return {"auth": "api", "api_key": "test-secret"}

        model = q.Model("claude", "anthropic", "claude-opus-5-5", auth="cli")
        error = urllib.error.HTTPError("https://api.anthropic.com", 401, "unauthorized", {}, None)
        with patch.object(q, "load_config", return_value=SimpleNamespace(panel=[model])), \
             patch.object(webapp.urllib.request, "urlopen", side_effect=error), \
             patch.object(connections, "save_claude") as save:
            response = asyncio.run(webapp.claude_connection(Request()))
            self.assertEqual(response.status_code, 400)
            self.assertNotIn(b"test-secret", response.body)
            save.assert_not_called()

    def test_cross_origin_write_rejected(self):
        request = SimpleNamespace(headers={"content-type": "application/json", "host": "quorum.test", "origin": "https://elsewhere.test"})
        response = asyncio.run(webapp.claude_connection(request))
        self.assertEqual(response.status_code, 403)

    def test_switch_preserves_model_config_and_private_key(self):
        with tempfile.TemporaryDirectory() as td:
            config = Path(td) / "config.toml"
            text = '[[panel]]\nname="claude"\nprovider="anthropic"\nmodel="claude-opus-5-5"\nauth="cli"\n'
            config.write_text(text)
            with patch.object(q, "CONFIG_PATH", config):
                connections.save_claude(config, "api", "test-secret")
                m = q.load_config().panel[0]
                self.assertEqual((m.auth, m.api_key, m.model), ("api", "test-secret", "claude-opus-5-5"))
                self.assertEqual(os.stat(connections.path_for(config)).st_mode & 0o777, 0o600)
                connections.save_claude(config, "cli")
                self.assertEqual(q.load_config().panel[0].api_key, "test-secret")
                connections.save_claude(config, "cli", remove=True)
                self.assertEqual(q.load_config().panel[0].api_key, "")
                self.assertEqual(config.read_text(), text)

    def test_api_tools_pause_and_usage(self):
        m = q.Model("claude", "anthropic", "claude-opus-5-5", api_key="test-secret")
        responses = [
            {"stop_reason": "pause_turn", "content": [{"type": "text", "text": "working"}],
             "usage": {"input_tokens": 100, "output_tokens": 20}},
            {"stop_reason": "end_turn", "content": [{"type": "text", "text": "answer"}],
             "usage": {"input_tokens": 150, "output_tokens": 30}},
        ]
        with patch.object(q, "_http_json", side_effect=responses) as http:
            self.assertEqual(q._api_anthropic(m, "question", True), "answer")
            self.assertEqual([t["name"] for t in http.call_args.args[1]["tools"]], ["web_search", "web_fetch"])
            self.assertEqual((m.tokens_in, m.tokens_out, m.tokens_est), (250, 50, False))

    def test_api_refusal_is_reported(self):
        m = q.Model("claude", "anthropic", "claude-opus-5-5", api_key="test-secret")
        refusal = {"stop_reason": "refusal", "content": [],
                   "stop_details": {"type": "refusal", "category": "cyber"},
                   "usage": {"input_tokens": 10, "output_tokens": 0}}
        with patch.object(q, "_http_json", return_value=refusal):
            with self.assertRaisesRegex(RuntimeError, "cyber"):
                q._api_anthropic(m, "question", True)

    def test_xai_responses_use_web_and_x_search(self):
        model = q.Model("grok", "xai", "grok-4.7", api_key="test-secret")
        response = {"status": "completed", "usage": {"input_tokens": 300, "output_tokens": 90,
                                                      "cost_in_usd_ticks": 2894140000,
                                                      "server_side_tool_usage_details": {
                                                          "web_search_calls": 5, "x_search_calls": 8,
                                                          "mcp_calls": 0}},
                    "output": [{"type": "message", "content": [
                        {"type": "output_text", "text": "research"}]}]}
        with patch.object(q, "_http_json", return_value=response) as http:
            self.assertEqual(q.call_model(model, "question", True), "research")
            self.assertEqual(http.call_args.args[0], "https://api.x.ai/v1/responses")
            self.assertEqual(http.call_args.args[1]["tools"], [{"type": "web_search"}, {"type": "x_search"}])
            self.assertEqual(http.call_args.args[2]["Authorization"], "Bearer test-secret")
            self.assertEqual(http.call_args.args[1]["max_turns"], q.XAI_MAX_TURNS)
        # Provider-reported cost (incl. searches) wins over the token estimate.
        self.assertAlmostEqual(q.estimated_api_cost(model), 0.289414)
        self.assertEqual(model.searches, {"web_search_calls": 5, "x_search_calls": 8})

    def test_grok_key_check_and_manual_mode(self):
        class Request:
            def __init__(self, body):
                self.body = body
            headers = {"content-type": "application/json", "host": "quorum.test", "origin": "https://quorum.test"}

            async def json(self):
                return self.body

        model = q.Model("grok", "xai", "grok-4.7", auth="api")
        with patch.object(q, "load_config", return_value=SimpleNamespace(panel=[model])), \
             patch.object(webapp.urllib.request, "urlopen") as upstream, \
             patch.object(connections, "save_claude") as save:
            upstream.return_value.__enter__.return_value.read.return_value = b'{}'
            response = asyncio.run(webapp.grok_connection(Request({"auth": "api", "api_key": "test-secret"})))
            self.assertEqual(response.status_code, 200)
            self.assertEqual(upstream.call_args.args[0].full_url, "https://api.x.ai/v1/models/grok-4.7")
            self.assertEqual(save.call_args.args[1:], ("api", "test-secret", False, "xai"))
            # Grok has no CLI: "cli" is rejected, "manual" saves without a key check.
            upstream.reset_mock()
            response = asyncio.run(webapp.grok_connection(Request({"auth": "cli"})))
            self.assertEqual(response.status_code, 400)
            response = asyncio.run(webapp.grok_connection(Request({"auth": "manual"})))
            self.assertEqual(response.status_code, 200)
            upstream.assert_not_called()

    def test_pipeline_runs_grok_when_no_research_pasted(self):
        claude = q.Model("claude", "anthropic", "claude-opus-5-5", auth="cli")
        gpt = q.Model("gpt", "openai", "gpt-6.1-sol", auth="cli")
        grok = q.Model("grok", "xai", "grok-4.7", api_key="test-secret")
        prompts = {}

        def fake_run(m, prompt, web_tools=False):
            prompts[m.provider] = prompts.get(m.provider, []) + [prompt]
            m.answer = f"{m.provider} answer"
            return m

        class Request:
            async def json(self):
                return {"question": "Что нового?"}

        async def collect():
            response = await webapp.api_pipeline(Request())
            return [json.loads(chunk[6:]) async for chunk in response.body_iterator][1:]

        with patch.object(q, "load_config", return_value=SimpleNamespace(panel=[claude, gpt, grok])), \
             patch.object(q, "run_model", side_effect=fake_run), \
             patch.object(webapp, "blockchain_evidence", return_value="none"):
            events = asyncio.run(collect())
        stages = [e["stage"] for e in events if e["type"] == "stage"]
        self.assertEqual(stages, ["research", "draft", "review", "final"])
        self.assertIn("Что нового?", prompts["xai"][0])
        self.assertIn("xai answer", prompts["anthropic"][0])
        self.assertTrue(events[-1]["ok"])

        # A page reload re-attaches to the stored run and replays the same log.
        stored = runs.recent()
        self.assertEqual([(r["status"], r["ok"], r["title"]) for r in stored], [("done", True, "Что нового?")])

        async def replay():
            response = await webapp.api_run_stream(stored[0]["id"])
            return [json.loads(chunk[6:]) async for chunk in response.body_iterator][1:]
        self.assertEqual(asyncio.run(replay()), events)
        self.assertEqual(os.stat(runs.runs_dir() / f"{stored[0]['id']}.json").st_mode & 0o777, 0o600)

    def test_revision_reuses_research_without_grok(self):
        parent = runs.new("Что нового?", "", "")
        parent.update(status="done", ok=True, events=[
            {"type": "stage", "stage": "research", "provider": "xai", "model": "grok-4.7",
             "answer": "grok packet", "error": "", "cost_usd": 0.25},
            {"type": "stage", "stage": "final", "provider": "anthropic", "model": "claude-opus-5-5",
             "answer": "old final", "error": ""},
            {"type": "done", "ok": True}])
        runs.save(parent)
        claude = q.Model("claude", "anthropic", "claude-opus-5-5", auth="cli")
        gpt = q.Model("gpt", "openai", "gpt-6.1-sol", auth="cli")
        grok = q.Model("grok", "xai", "grok-4.7", api_key="test-secret")
        prompts = []

        def fake_run(m, prompt, web_tools=False):
            prompts.append((m.provider, prompt))
            m.answer = f"{m.provider} answer"
            return m

        class Request:
            def __init__(self, text):
                self.text = text

            async def json(self):
                return {"revision": self.text}

        async def collect():
            response = await webapp.api_revise(parent["id"], Request("сократи"))
            return [json.loads(chunk[6:]) async for chunk in response.body_iterator][1:]

        with patch.object(q, "load_config", return_value=SimpleNamespace(panel=[claude, gpt, grok])), \
             patch.object(q, "run_model", side_effect=fake_run), \
             patch.object(webapp, "blockchain_evidence", return_value="none"):
            events = asyncio.run(collect())
            rejected = asyncio.run(webapp.api_revise(parent["id"], Request(" ")))
        self.assertEqual(rejected.status_code, 400)
        self.assertEqual([p for p, _ in prompts], ["anthropic", "openai", "anthropic"])  # no xai call
        self.assertIn("grok packet", prompts[0][1])
        self.assertIn("сократи", prompts[0][1])
        self.assertIn("old final", prompts[0][1])
        self.assertEqual(events[0]["type"], "revision")
        self.assertIsNone(events[1]["cost_usd"])  # reused research is not billed again
        self.assertTrue(events[-1]["ok"])
        child = [r for r in runs.recent() if r["parent"]]
        self.assertEqual(child[0]["parent"], parent["id"])

    def test_history_title_shortens_links(self):
        self.assertEqual(runs.title("https://x.com/KobeissiLetter/status/2111679803560796451\n\nПризывы к «Черному понедельнику»"),
                         "KobeissiLetter · Призывы к «Черному понедельнику»")
        self.assertEqual(runs.title("что тут? https://x.com/xai/status/1"), "что тут? xai")
        self.assertEqual(runs.title("см. https://www.coindesk.com/markets/2026/10/18/x что это?"), "см. coindesk.com что это?")

    def test_history_title_uses_grok_topic(self):
        research = "**Тема:** Призывы к «Чёрному понедельнику»\n\n## Краткая карта исследования\n..."
        self.assertEqual(runs.topic(research), "Призывы к «Чёрному понедельнику»")
        self.assertEqual(runs.topic("## Без темы"), "")
        self.assertEqual(runs.title("https://x.com/KobeissiLetter/status/1\nдлинный текст", "Призывы к «Чёрному понедельнику»"),
                         "KobeissiLetter · Призывы к «Чёрному понедельнику»")
        self.assertEqual(runs.title("Что нового у xAI?", "Новости xAI"), "Новости xAI")

    def test_run_interrupted_by_restart_is_reported(self):
        run = runs.new("вопрос", "", "")
        runs.save(run)  # status "running", but no live task owns it
        restored = webapp._stored_run(run["id"])
        self.assertEqual(restored["status"], "done")
        self.assertIn("перезапуском", restored["events"][-1]["message"])
        self.assertIsNone(webapp._stored_run("../../etc/passwd"))

    def test_pipeline_without_grok_key_asks_for_it(self):
        claude = q.Model("claude", "anthropic", "claude-opus-5-5", auth="cli")
        gpt = q.Model("gpt", "openai", "gpt-6.1-sol", auth="cli")
        grok = q.Model("grok", "xai", "grok-4.7")

        class Request:
            async def json(self):
                return {"question": "Что нового?"}

        async def collect():
            response = await webapp.api_pipeline(Request())
            return [json.loads(chunk[6:]) async for chunk in response.body_iterator][1:]

        with patch.object(q, "load_config", return_value=SimpleNamespace(panel=[claude, gpt, grok])), \
             patch.object(q, "run_model") as run:
            events = asyncio.run(collect())
        self.assertEqual(events[0]["type"], "error")
        self.assertIn("xAI API key", events[0]["message"])
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
