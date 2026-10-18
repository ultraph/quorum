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
import webapp


class ConnectionTests(unittest.TestCase):
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
        response = {"status": "completed", "usage": {"input_tokens": 300, "output_tokens": 90},
                    "output": [{"type": "message", "content": [
                        {"type": "output_text", "text": "research"}]}]}
        with patch.object(q, "_http_json", return_value=response) as http:
            self.assertEqual(q.call_model(model, "question", True), "research")
            self.assertEqual(http.call_args.args[0], "https://api.x.ai/v1/responses")
            self.assertEqual(http.call_args.args[1]["tools"], [{"type": "web_search"}, {"type": "x_search"}])
            self.assertEqual(http.call_args.args[2]["Authorization"], "Bearer test-secret")
        self.assertAlmostEqual(q.estimated_api_cost(model), (300 * 2.00 + 90 * 6.00) / 1_000_000)

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
            return [json.loads(chunk[6:]) async for chunk in response.body_iterator]

        with patch.object(q, "load_config", return_value=SimpleNamespace(panel=[claude, gpt, grok])), \
             patch.object(q, "run_model", side_effect=fake_run), \
             patch.object(webapp, "blockchain_evidence", return_value="none"):
            events = asyncio.run(collect())
        stages = [e["stage"] for e in events if e["type"] == "stage"]
        self.assertEqual(stages, ["research", "draft", "review", "final"])
        self.assertIn("Что нового?", prompts["xai"][0])
        self.assertIn("xai answer", prompts["anthropic"][0])
        self.assertTrue(events[-1]["ok"])

    def test_pipeline_without_grok_key_asks_for_it(self):
        claude = q.Model("claude", "anthropic", "claude-opus-5-5", auth="cli")
        gpt = q.Model("gpt", "openai", "gpt-6.1-sol", auth="cli")
        grok = q.Model("grok", "xai", "grok-4.7")

        class Request:
            async def json(self):
                return {"question": "Что нового?"}

        async def collect():
            response = await webapp.api_pipeline(Request())
            return [json.loads(chunk[6:]) async for chunk in response.body_iterator]

        with patch.object(q, "load_config", return_value=SimpleNamespace(panel=[claude, gpt, grok])), \
             patch.object(q, "run_model") as run:
            events = asyncio.run(collect())
        self.assertEqual(events[0]["type"], "error")
        self.assertIn("xAI API key", events[0]["message"])
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
