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


if __name__ == "__main__":
    unittest.main()
