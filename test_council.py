"""Duel / council modes: pure logic and the whole resumable flow with fake
models (no real model calls). Run: python -m unittest test_council -v"""
import asyncio
import json
import os
import random
import re
import tempfile
import unittest
from unittest.mock import patch

import council
import quorum as q
import webapp


class LogicTests(unittest.TestCase):
    def test_labels_are_a_permutation(self):
        labels = council.assign_labels(council.AUTHORS["council"], random.Random(1))
        self.assertEqual(sorted(labels.values()), list("ABCDE"))

    def test_blind_block_excludes_own_answer(self):
        labels = {"claude": "B", "codex": "A", "deepseek": "C"}
        block, shown = council.blind_block({"claude": "c", "codex": "x", "deepseek": "d"}, labels, exclude="claude")
        self.assertEqual(shown, ["A", "C"])
        self.assertNotIn("ANSWER B", block)
        self.assertLess(block.index("ANSWER A"), block.index("ANSWER C"))

    def test_parse_winner_takes_last_valid(self):
        self.assertEqual(council.parse_winner("x\nWINNER: A\n...\n**WINNER: B**", {"A", "B"}), "B")
        self.assertIsNone(council.parse_winner("WINNER: C", {"A", "B"}))
        self.assertIsNone(council.parse_winner("no verdict", {"A", "B"}))

    def test_parse_review_requires_exact_labels(self):
        text = "...\nRANKING: C > A > D\nSCORES: A=7, C=9, D=4"
        self.assertEqual(council.parse_review(text, ["A", "C", "D"]),
                         {"ranking": ["C", "A", "D"], "scores": {"A": 7.0, "C": 9.0, "D": 4.0}})
        self.assertIsNone(council.parse_review(text, ["A", "C", "D", "E"]))
        self.assertIsNone(council.parse_review("RANKING: C > A", ["A", "C", "D"]))

    def test_aggregate_normalizes_unequal_votes(self):
        reviews = {"r1": {"ranking": ["A", "B", "C"], "scores": {}},
                   "r2": {"ranking": ["B", "A"], "scores": {}}}
        rows = council.aggregate(reviews, ["A", "B", "C"])
        self.assertEqual([(r["label"], r["borda"]) for r in rows], [("B", 0.75), ("A", 0.5), ("C", 0.0)])
        self.assertEqual(rows[2]["votes"], 1)

    def test_aggregate_tie_broken_by_mean_score(self):
        reviews = {"r1": {"ranking": ["A", "B"], "scores": {"A": 6, "B": 9}},
                   "r2": {"ranking": ["B", "A"], "scores": {"A": 6, "B": 8}}}
        self.assertEqual([r["label"] for r in council.aggregate(reviews, ["A", "B"])], ["B", "A"])

    def test_synthesizer_never_the_winner(self):
        self.assertEqual(council.synthesizer("claude", ["claude", "codex"]), "codex")
        self.assertEqual(council.synthesizer("grok", ["grok", "deepseek", "claude"]), "deepseek")
        self.assertEqual(council.synthesizer("codex", ["codex", "gemini", "claude"]), "claude")


def fake_config():
    return q.Config(panel=[
        q.Model(name="claude", provider="anthropic", model="claude-opus-5-5", auth="cli"),
        q.Model(name="gpt", provider="openai", model="gpt-6.1-sol", auth="cli"),
        q.Model(name="deepseek-pro", provider="deepseek", model="deepseek-v4-pro", api_key="x"),
    ])


def fake_run_model(m, prompt, web=False):
    """Deterministic stand-in: authors answer, judges prefer the first label."""
    m.tokens_in, m.tokens_out = 1000, 500
    if "impartial judge" in prompt:
        m.answer = "comparison\nWINNER: A"
    elif "blind peer reviewer" in prompt:
        shown = re.findall(r"=== ANSWER ([A-Z]) ===", prompt)
        m.answer = "review\nRANKING: " + " > ".join(shown) + "\nSCORES: " + ", ".join(f"{x}=7" for x in shown)
    elif "FINAL answer" in prompt:
        m.answer = f"final by {m.name}"
    elif "critic whose job is to disagree" in prompt:
        m.answer = f"revised by {m.name}"
    else:
        m.answer = f"answer by {m.name}"
    return m


class Req:
    def __init__(self, body):
        self.body = body

    async def json(self):
        return self.body


class FlowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.env = patch.dict(os.environ, {"QUORUM_RUNS_DIR": self.tmp.name})
        self.env.start()
        self.patches = [patch.object(q, "load_config", fake_config),
                        patch.object(q, "run_model", side_effect=fake_run_model)]
        self.calls = self.patches[1].start()
        self.patches[0].start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.env.stop()
        self.tmp.cleanup()

    def new_run(self, mode, judge="deepseek"):
        run = webapp.runs.new("Стоит ли X?", "", "")
        run.update(mode=mode, web=True, judge=judge, attachments=[], results={}, skipped=[],
                   requested=[], pending=[])
        webapp.LIVE[run["id"]] = run
        return run

    async def manual(self, run, body):
        resp = await webapp.api_manual(run["id"], Req(body))
        await asyncio.gather(*list(webapp.TASKS))
        return resp.status_code, json.loads(resp.body)

    def test_duel_deepseek_then_optional_critic(self):
        async def go():
            run = self.new_run("duel")
            await webapp._execute_council(run)
            self.assertEqual(run["status"], "waiting")  # optional Grok critic
            self.assertTrue(run["ok"])
            winner = {v: k for k, v in run["labels"].items()}["A"]
            synth = run["synth_model"]
            self.assertNotEqual(winner, synth)
            self.assertEqual(run["results"]["synth"]["answer"],
                             f"final by {'claude' if synth == 'claude' else 'gpt'}")
            code, _ = await self.manual(run, {"key": "critic", "text": "критика"})
            self.assertEqual(code, 200)
            run = webapp.runs.load(run["id"])
            self.assertEqual(run["status"], "waiting")  # «учесть критику» offered
            code, _ = await self.manual(run, {"key": "apply", "go": True})
            run = webapp.runs.load(run["id"])
            self.assertEqual(run["status"], "done")
            self.assertTrue(run["results"]["apply"]["answer"].startswith("revised by"))
        asyncio.run(go())

    def test_duel_gemini_judge_validates_paste(self):
        async def go():
            run = self.new_run("duel", judge="gemini")
            await webapp._execute_council(run)
            self.assertEqual(run["status"], "waiting")
            self.assertIn("judge", run["pending"])
            code, data = await self.manual(run, {"key": "judge", "text": "без вердикта"})
            self.assertEqual(code, 400)
            code, _ = await self.manual(run, {"key": "judge", "text": "ok\nWINNER: B"})
            self.assertEqual(code, 200)
            run = webapp.runs.load(run["id"])
            self.assertTrue(run["final_done"])
            self.assertEqual(run["ranking"][0]["label"], "B")
        asyncio.run(go())

    def test_council_manual_authors_and_no_self_review(self):
        async def go():
            run = self.new_run("council")
            await webapp._execute_council(run)
            self.assertEqual(run["status"], "waiting")
            self.assertEqual(sorted(run["pending"]), ["author:gemini", "author:grok"])
            await self.manual(run, {"key": "author:grok", "text": "answer by grok"})
            run = webapp.runs.load(run["id"])
            self.assertEqual(run["status"], "waiting")  # Gemini still missing
            await self.manual(run, {"key": "author:gemini", "skip": True})
            run = webapp.runs.load(run["id"])
            self.assertIn("ranking", run)
            for call in self.calls.call_args_list:
                m, prompt = call.args[0], call.args[1]
                if "blind peer reviewer" in prompt:
                    own = run["labels"][{"claude": "claude", "gpt": "codex", "deepseek-pro": "deepseek"}[m.name]]
                    self.assertNotIn(f"=== ANSWER {own} ===", prompt)
                    self.assertEqual(len(re.findall(r"=== ANSWER", prompt)), 3)  # 4 answers − own
            winner = run["ranking"][0]["who"]
            self.assertNotEqual(council.NAMES[run["synth_model"]], winner)
            await self.manual(run, {"key": "critic", "skip": True})
            run = webapp.runs.load(run["id"])
            self.assertEqual(run["status"], "done")
            self.assertTrue(run["ok"])
        asyncio.run(go())

    def test_failed_author_is_retried_alone(self):
        def flaky(m, prompt, web=False):
            if m.name == "gpt" and not flaky.failed:
                flaky.failed = True
                m.error = "boom"
                return m
            return fake_run_model(m, prompt, web)
        flaky.failed = False
        self.calls.side_effect = flaky

        async def go():
            run = self.new_run("duel")
            await webapp._execute_council(run)
            self.assertEqual((run["status"], run["ok"]), ("done", False))
            before = self.calls.call_count
            code, data = await self.manual(run, {"retry": True})
            self.assertEqual((code, data["started"]), (200, True))
            run = webapp.runs.load(run["id"])
            self.assertTrue(run["final_done"])
            names = [c.args[0].name for c in self.calls.call_args_list[before:]]
            self.assertEqual(names[0], "gpt")  # Claude's answer was not recomputed
            self.assertNotIn("claude", names[:1])
        asyncio.run(go())


if __name__ == "__main__":
    unittest.main()
