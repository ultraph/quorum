"""Blind multi-model modes: «Дуэль» (duel) and «Совет» (council).

Pure logic only — prompts, blind labels, parsing of judge/peer verdicts and
the Borda aggregation — so it is testable without models. The async
orchestration (model calls, pauses for manual pastes) lives in webapp.py.

Participants are keyed by role, not by model id:
  claude   — Claude, subscription CLI or API (automatic)
  codex    — GPT/Codex, subscription CLI or API (automatic)
  deepseek — DeepSeek V4 Pro API (automatic)
  grok     — Grok, pasted by hand from the Grok chat
  gemini   — Gemini, pasted by hand from the Gemini chat
"""
import random
import re

MODES = ("duel", "council")
AUTO = ("claude", "codex", "deepseek")
MANUAL = ("grok", "gemini")
NAMES = {"claude": "Claude", "codex": "GPT/Codex", "deepseek": "DeepSeek",
         "grok": "Grok", "gemini": "Gemini"}
# Who writes in each mode. In the duel only the two subscription models write.
AUTHORS = {"duel": ("claude", "codex"), "council": ("claude", "codex", "deepseek", "grok", "gemini")}
# Peer reviewers in the council: automatic models only (owner's decision
# 20.11.2026 — manual Grok/Gemini write but do not grade, fewer pastes).
REVIEWERS = ("claude", "codex", "deepseek")


def assign_labels(authors, rng=None) -> dict:
    """Random blind letters per author; stored in the run for reproducibility."""
    rng = rng or random.SystemRandom()
    letters = [chr(ord("A") + i) for i in range(len(authors))]
    rng.shuffle(letters)
    return dict(zip(authors, letters))


def blind_block(answers: dict, labels: dict, exclude: str = "") -> tuple[str, list[str]]:
    """Answers under their letters, alphabetical (= random author order);
    `exclude` drops the reviewer's own answer so nobody grades themselves."""
    shown = sorted((labels[k], text) for k, text in answers.items() if k != exclude and text)
    block = "\n\n".join(f"=== ANSWER {letter} ===\n{text}" for letter, text in shown)
    return block, [letter for letter, _ in shown]


AUTHOR = """\
You are one of several independent experts answering the same request. You
work alone: you will not see the other answers, and the answers will be
compared blind, so do not mention who or what you are.

Give your own best, complete answer for a person who has to decide or act on
it. Lead with the recommendation or the direct answer, then the reasoning,
the decision-relevant numbers, the main risks and what would change your
answer. If something essential is unknown, say exactly what and how it
changes the conclusion — do not fill gaps with guesses.
{web_rule}
Be concise but complete (up to about 700 words). Write in the user's
language. Write every EVM address in full (`0x` + 40 hexadecimal characters).
Today is {today}.

=== REQUEST ===
{question}
"""

WEB_ON = "Use web search to check facts and fresh data where it matters; cite URLs for key facts."
WEB_OFF = "Answer from your own knowledge; mark anything that may be outdated."
WEB_MANUAL = "Use web search (and X, if relevant) to check facts and fresh data; cite URLs for key facts."

CRITERIA = ("correctness (facts, numbers, logic), completeness on what matters for the decision, "
            "practical usefulness (clear recommendation, next step) and honesty about risks and unknowns")

DUEL_JUDGE = """\
You are an impartial judge. Two anonymous answers (A and B) were written
independently to the same request. Compare them on: {criteria}. Length and
style are not merits by themselves.

Reply in the user's language with:
1. A short comparison per criterion (who is better and why, concretely).
2. "Взять у проигравшего" — the specific strong points, facts or risks of the
   losing answer that the final version must keep (or "ничего").
3. Errors to fix in the winning answer, if any.

The LAST line must be exactly `WINNER: A` or `WINNER: B`.

=== REQUEST ===
{question}

{answers}
"""

PEER_REVIEW = """\
You are a blind peer reviewer. Several anonymous answers were written
independently to the same request. Grade each answer on: {criteria}.
Length and style are not merits by themselves. Do not guess who wrote what.

Reply in the user's language with, for each answer, a score from 1 to 10 and
two or three lines: its strongest point and its most serious flaw (errors
first). Then a short list of the best ideas from any answer that a final
version should combine.

End with exactly these two lines, covering every answer shown ({labels}):
`RANKING: <best> > <next> > ... > <worst>`
`SCORES: <letter>=<1-10>, <letter>=<1-10>, ...`

=== REQUEST ===
{question}

{answers}
"""

SYNTH = """\
You write the FINAL answer the user will read. Several experts answered the
request independently and blind; the answers were then judged. The winning
answer is the base. Keep its structure and conclusions unless the judging
shows they are wrong; add the strong points of the other answers named in the
judging; fix the errors the judging found. Do not invent new facts; keep the
URLs that support key facts.

Do not mention the process, the experts, labels, judges or models — write as
one voice. Reply in the user's language, at most about 400 words:
recommendation or direct answer first, then why, the key numbers, risks and
unknowns, and the concrete next step. Write every EVM address in full
(`0x` + 40 hexadecimal characters). Today is {today}.

=== REQUEST ===
{question}

=== WINNING ANSWER ({winner}) ===
{winner_text}

=== OTHER ANSWERS ===
{others}

=== JUDGING ===
{judging}
"""

CRITIC = """\
Your job is to DISAGREE. Below is a final answer another AI prepared for a
person who will act on it. Tear it apart before the person sees it: find
factual errors, wrong numbers, weak or hidden assumptions, missing risks,
strong counterarguments and better alternatives. Check facts on the web and
in X where it matters and give URLs. Do not praise and do not rewrite the
answer; if a point is actually solid, skip it.

Reply in the user's language. A numbered list sorted by severity, each item
marked [критично], [важно] or [мелочь], with a one-line fix. Last line:
`ВЕРДИКТ: выдерживает | нужны правки | неверно`. Today is {today}.

=== REQUEST ===
{question}

=== FINAL ANSWER TO ATTACK ===
{final}
"""

APPLY = """\
You wrote the final answer below. A critic whose job is to disagree attacked
it. Revise the final answer: accept only the criticism that is correct and
matters (check it; a critic can be wrong), keep everything else.
- New facts the critic found with a source (names, numbers, quotes from a
  linked page) are evidence: carry the specific fact into the answer
  explicitly, with its URL, not a vague mention.
- Never drop a check, step or fact just because the critic calls it
  impractical or unlikely to be done: first ask who would do it (the user's
  own team may do what the end client would not), then adjust it. Remove an
  item only if it is wrong.
Same rules:
the user's language, at most about 400 words, recommendation first, full EVM
addresses, no mention of the process or models. Finish with a short section
"Что изменилось после критики" (2–4 bullets; "ничего существенного" if the
criticism did not hold, with one line why). Today is {today}.

=== REQUEST ===
{question}

=== YOUR FINAL ANSWER ===
{final}

=== CRITIQUE ===
{critique}
"""

COUNCIL_CHAT = """\
You are {name}. A blind multi-model review ({mode}) has finished on the
request below: independent answers, blind judging, a final answer{critic_note}.
The user now continues the conversation with you alone. Answer the user's
latest message, grounded in the material below first; use web tools only for
what it does not cover, and cite the URL. Do not repeat the whole final
answer. Write every EVM address in full. Write in the user's language.
Today is {today}.

=== REQUEST ===
{question}

{material}

=== CONVERSATION SO FAR ===
{history}
"""


_WINNER = re.compile(r"WINNER\s*[:：]\s*\**\s*([A-Z])\b", re.I)
_RANKING = re.compile(r"RANKING\s*[:：]\s*\**\s*([A-Z](?:\s*[>,]\s*[A-Z])*)", re.I)
_SCORES = re.compile(r"SCORES\s*[:：]\s*(.+)", re.I)
_SCORE_ITEM = re.compile(r"\b([A-Z])\s*[=:]\s*(\d+(?:[.,]\d+)?)", re.I)


def parse_winner(text: str, allowed) -> str | None:
    found = _WINNER.findall(text or "")
    letter = found[-1].upper() if found else None
    return letter if letter in allowed else None


def parse_review(text: str, shown) -> dict | None:
    """{"ranking": [letters best→worst], "scores": {letter: float}} or None
    when the ranking does not cover exactly the answers shown."""
    found = _RANKING.findall(text or "")
    if not found:
        return None
    ranking = [x.upper() for x in re.findall(r"[A-Za-z]", found[-1])]
    if sorted(ranking) != sorted(shown):
        return None
    scores = {}
    line = _SCORES.findall(text or "")
    if line:
        for letter, value in _SCORE_ITEM.findall(line[-1]):
            if letter.upper() in shown:
                scores[letter.upper()] = min(10.0, max(0.0, float(value.replace(",", "."))))
    return {"ranking": ranking, "scores": scores}


def aggregate(reviews: dict, letters) -> list[dict]:
    """Borda over blind rankings. Reviewers never see their own answer, so
    answers get different numbers of votes: points are normalized to 0..1 per
    ranking (best = 1, worst = 0) and averaged. Ties → mean score, then the
    number of first places. Returns rows best first."""
    rows = {x: {"label": x, "points": [], "scores": [], "firsts": 0} for x in letters}
    for review in reviews.values():
        ranking = review["ranking"]
        n = len(ranking)
        for pos, letter in enumerate(ranking):
            if letter in rows:
                rows[letter]["points"].append(1.0 if n == 1 else (n - 1 - pos) / (n - 1))
                if pos == 0:
                    rows[letter]["firsts"] += 1
        for letter, score in review.get("scores", {}).items():
            if letter in rows:
                rows[letter]["scores"].append(score)
    out = []
    for row in rows.values():
        votes = len(row["points"])
        out.append({"label": row["label"], "votes": votes, "firsts": row["firsts"],
                    "borda": round(sum(row["points"]) / votes, 3) if votes else 0.0,
                    "score": round(sum(row["scores"]) / len(row["scores"]), 2) if row["scores"] else None})
    out.sort(key=lambda r: (-r["borda"], -(r["score"] or 0), -r["firsts"], r["label"]))
    return out


def synthesizer(winner: str, order, mode: str = "duel") -> str:
    """Who writes the final; never the author of the winning answer.
    Duel: the losing author (`order` is winner, loser). Council: Claude, or
    Codex when Claude won (owner's decision 20.11.2026: the best-ranked
    automatic author was DeepSeek, whose final re-invented unsupported
    percentages and dropped the sources)."""
    if mode == "council":
        return "codex" if winner == "claude" else "claude"
    for key in order:
        if key != winner and key in AUTO:
            return key
    return next(k for k in AUTO if k != winner)


def author_prompt(question: str, today: str, key: str, web: bool) -> str:
    if key in MANUAL:
        rule = WEB_MANUAL
    else:
        rule = WEB_ON if web and key != "deepseek" else WEB_OFF
    return AUTHOR.format(question=question, today=today, web_rule=rule)


def duel_judge_prompt(question: str, answers: dict, labels: dict) -> str:
    block, _ = blind_block(answers, labels)
    return DUEL_JUDGE.format(criteria=CRITERIA, question=question, answers=block)


def peer_review_prompt(question: str, answers: dict, labels: dict, reviewer: str) -> tuple[str, list[str]]:
    block, shown = blind_block(answers, labels, exclude=reviewer)
    return PEER_REVIEW.format(criteria=CRITERIA, question=question, answers=block,
                              labels=", ".join(shown)), shown


def synth_prompt(question: str, today: str, answers: dict, labels: dict, winner: str, judging: str) -> str:
    others, _ = blind_block({k: v for k, v in answers.items() if k != winner}, labels)
    return SYNTH.format(question=question, today=today, winner=f"answer {labels[winner]}",
                        winner_text=answers[winner], others=others or "(none)", judging=judging)


def critic_prompt(question: str, today: str, final: str) -> str:
    return CRITIC.format(question=question, today=today, final=final)


def apply_prompt(question: str, today: str, final: str, critique: str) -> str:
    return APPLY.format(question=question, today=today, final=final, critique=critique)
