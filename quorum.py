#!/usr/bin/env python3
"""quorum — ask a panel of LLMs one question and get a judge's synthesis.

Pick which models sit on the panel and who judges (checkboxes or flags).
Each model can be reached two ways, per your config:

  auth = "api"  → official API with your key   (Anthropic / OpenAI / Gemini / DeepSeek)
  auth = "cli"  → your local CLI login         (claude -p / codex exec)

Zero third-party dependencies — standard library only. HTTP via urllib,
checkboxes via curses (with a plain numbered fallback when curses is absent).

Config: ~/.config/quorum/config.toml  (see config.example.toml)
"""

from __future__ import annotations

import argparse
import base64
import getpass
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

CONFIG_PATH = Path(os.environ.get(
    "QUORUM_CONFIG", Path.home() / ".config" / "quorum" / "config.toml"))

_TTY = sys.stdout.isatty()
def _c(code: str) -> str:
    return code if _TTY else ""
BOLD, DIM, RED, GRN, YEL, CYA, RST = (
    _c("\033[1m"), _c("\033[2m"), _c("\033[31m"), _c("\033[32m"),
    _c("\033[33m"), _c("\033[36m"), _c("\033[0m"))


# --- model definitions ------------------------------------------------------

@dataclass
class Model:
    name: str                # short label shown to the user
    provider: str            # anthropic | openai | gemini | deepseek
    model: str               # provider model id
    auth: str = "api"        # api | cli
    api_key: str = ""        # resolved key (api mode)
    cli_bin: str = ""        # binary name/path (cli mode)
    enabled: bool = True
    # result:
    answer: str = ""
    error: str = ""
    seconds: float = 0.0


@dataclass
class Config:
    panel: list[Model] = field(default_factory=list)
    judge: Model | None = None
    judge_enabled: bool = True


def _resolve_key(raw: str) -> str:
    """Support "env:VAR" (read env), otherwise literal."""
    if raw.startswith("env:"):
        return os.environ.get(raw[4:], "")
    return raw


def _model_from_table(t: dict) -> Model:
    return Model(
        name=t["name"], provider=t["provider"], model=t["model"],
        auth=t.get("auth", "api"),
        api_key=_resolve_key(t.get("api_key", "")),
        cli_bin=t.get("cli_bin", ""),
        enabled=t.get("enabled", True),
    )


# --- TOML loading -----------------------------------------------------------
# Python 3.11+ ships tomllib. On older interpreters (e.g. Ubuntu 22.04 / Mint 21
# = Python 3.10) it's absent, so fall back to a tiny reader that handles the flat
# subset quorum writes and documents. Keeps the "stdlib only, zero deps" promise.

try:
    import tomllib  # Python 3.11+

    def _load_toml(f) -> dict:
        return tomllib.load(f)
except ModuleNotFoundError:
    def _load_toml(f) -> dict:
        return _mini_toml(f.read().decode("utf-8"))


def _strip_inline_comment(s: str) -> str:
    """Drop a trailing `# comment`, but not a `#` inside a quoted string."""
    out, quote, esc = [], "", False
    for ch in s:
        if esc:                       # previous char was a backslash (basic str)
            out.append(ch); esc = False; continue
        if quote == '"' and ch == "\\":
            out.append(ch); esc = True; continue
        if quote:
            out.append(ch)
            if ch == quote:
                quote = ""
        elif ch in ('"', "'"):
            quote = ch; out.append(ch)
        elif ch == "#":
            break
        else:
            out.append(ch)
    return "".join(out).strip()


def _mini_value(s: str, lineno: int):
    s = _strip_inline_comment(s)
    if not s:
        raise ValueError(f"line {lineno}: empty value")
    if s[0] == '"':                   # basic string — JSON escapes match TOML's
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            raise ValueError(f"line {lineno}: bad string: {s!r}")
    if s[0] == "'":                   # literal string — no escapes
        if len(s) < 2 or s[-1] != "'":
            raise ValueError(f"line {lineno}: unterminated string: {s!r}")
        return s[1:-1]
    if s == "true":
        return True
    if s == "false":
        return False
    try:
        return int(s)
    except ValueError:
        raise ValueError(f"line {lineno}: cannot parse value: {s!r}")


def _mini_toml(text: str) -> dict:
    """Minimal TOML reader for quorum's own flat config shape.

    Supports exactly what quorum writes/documents: [table], [[array.of.tables]],
    key = "string" | 'literal' | true | false | int, full-line and inline `#`
    comments, and blank lines. Not a general TOML parser — used only as a
    fallback when stdlib tomllib is unavailable (Python < 3.11).
    """
    root: dict = {}
    cur: dict = root
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[[") and line.endswith("]]"):
            key = line[2:-2].strip()
            arr = root.setdefault(key, [])
            if not isinstance(arr, list):
                raise ValueError(f"line {lineno}: {key!r} is not an array of tables")
            cur = {}
            arr.append(cur)
        elif line.startswith("[") and line.endswith("]"):
            key = line[1:-1].strip()
            tbl = root.setdefault(key, {})
            if not isinstance(tbl, dict):
                raise ValueError(f"line {lineno}: {key!r} is not a table")
            cur = tbl
        elif "=" in line:
            k, _, v = line.partition("=")
            cur[k.strip()] = _mini_value(v, lineno)
        else:
            raise ValueError(f"line {lineno}: expected 'key = value', got {line!r}")
    return root


def load_config() -> Config:
    if not CONFIG_PATH.exists():
        sys.exit(f"{RED}No config:{RST} {CONFIG_PATH}\n"
                 f"Copy config.example.toml there and fill it in.")
    with open(CONFIG_PATH, "rb") as f:
        raw = _load_toml(f)
    panel = [_model_from_table(t) for t in raw.get("panel", [])]
    jt = raw.get("judge", {})
    judge_enabled = jt.get("enabled", True)
    judge = None
    if "use" in jt:  # reuse a panelist by name
        judge = next((m for m in panel if m.name == jt["use"]), None)
        if judge is None:
            sys.exit(f"{RED}judge.use = '{jt['use']}' matches no panelist.{RST}")
    elif "provider" in jt:  # inline judge definition
        judge = _model_from_table({**jt, "name": jt.get("name", "judge")})
    return Config(panel=panel, judge=judge, judge_enabled=judge_enabled)


# --- providers --------------------------------------------------------------

def _http_json(url: str, payload: dict, headers: dict, timeout: int = 180) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _api_anthropic(m: Model, prompt: str) -> str:
    if not m.api_key:
        raise RuntimeError("no API key (set api_key, e.g. env:ANTHROPIC_API_KEY)")
    out = _http_json(
        "https://api.anthropic.com/v1/messages",
        {"model": m.model, "max_tokens": 4096,
         "messages": [{"role": "user", "content": prompt}]},
        {"x-api-key": m.api_key, "anthropic-version": "2023-06-01",
         "content-type": "application/json"})
    return "".join(b.get("text", "") for b in out["content"]
                   if b.get("type") == "text").strip()


def _api_openai_compatible(m: Model, prompt: str, base: str) -> str:
    if not m.api_key:
        raise RuntimeError("no API key")
    out = _http_json(
        base.rstrip("/") + "/chat/completions",
        {"model": m.model, "messages": [{"role": "user", "content": prompt}],
         "stream": False},
        {"Authorization": f"Bearer {m.api_key}", "Content-Type": "application/json"})
    return out["choices"][0]["message"]["content"].strip()


def _api_gemini(m: Model, prompt: str) -> str:
    if not m.api_key:
        raise RuntimeError("no API key")
    out = _http_json(
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{m.model}:generateContent?key={m.api_key}",
        {"contents": [{"parts": [{"text": prompt}]}]},
        {"Content-Type": "application/json"})
    return "".join(p.get("text", "")
                   for p in out["candidates"][0]["content"]["parts"]).strip()


def _find_bin(name: str, explicit: str = "") -> str:
    """Resolve a CLI binary without relying on the caller's PATH (nvm-aware)."""
    if explicit:
        return os.path.expanduser(explicit)
    if (found := shutil.which(name)):
        return found
    node = Path.home() / ".nvm" / "versions" / "node"
    cands = sorted(node.glob(f"*/bin/{name}"))
    return str(cands[-1]) if cands else name


def _bin_ok(binp: str) -> bool:
    return bool(shutil.which(binp)) or os.path.exists(binp)


def _cli_claude(m: Model, prompt: str) -> str:
    binp = _find_bin("claude", m.cli_bin)
    if not _bin_ok(binp):
        raise RuntimeError("claude CLI not found — install it and log in "
                           "(run `claude`, then /login)")
    p = subprocess.run([binp, "-p", "--model", m.model], input=prompt,
                       capture_output=True, text=True, timeout=420)
    if p.returncode != 0:
        raise RuntimeError((p.stderr or p.stdout or "claude failed").strip()[:400])
    return p.stdout.strip()


def _cli_codex(m: Model, prompt: str) -> str:
    binp = _find_bin("codex", m.cli_bin)
    if not _bin_ok(binp):
        raise RuntimeError("codex CLI not found — install it and log in (`codex login`)")
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "last.txt"
        p = subprocess.run(
            [binp, "exec", "--model", m.model, "--sandbox", "read-only",
             "--skip-git-repo-check", "--cd", td, "-o", str(out), "-"],
            input=prompt, capture_output=True, text=True, timeout=420)
        if out.exists() and (txt := out.read_text().strip()):
            return txt
        if p.returncode != 0:
            raise RuntimeError((p.stderr or p.stdout or "codex failed").strip()[:400])
        return p.stdout.strip()


def call_model(m: Model, prompt: str) -> str:
    if m.auth == "cli":
        if m.provider == "anthropic":
            return _cli_claude(m, prompt)
        if m.provider == "openai":
            return _cli_codex(m, prompt)
        raise RuntimeError(f"auth=cli unsupported for provider '{m.provider}'")
    # auth == api
    if m.provider == "anthropic":
        return _api_anthropic(m, prompt)
    if m.provider == "openai":
        return _api_openai_compatible(m, prompt, "https://api.openai.com/v1")
    if m.provider == "deepseek":
        return _api_openai_compatible(m, prompt, "https://api.deepseek.com")
    if m.provider == "gemini":
        return _api_gemini(m, prompt)
    raise RuntimeError(f"unknown provider '{m.provider}'")


def run_model(m: Model, prompt: str) -> Model:
    t0 = time.time()
    try:
        m.answer = call_model(m, prompt)
    except urllib.error.HTTPError as e:
        m.error = f"HTTP {e.code}: {e.read().decode(errors='replace')[:200]}"
    except Exception as e:  # noqa: BLE001
        m.error = str(e)[:400]
    m.seconds = time.time() - t0
    return m


# --- judge ------------------------------------------------------------------

JUDGE_PROMPT = """\
You are the judge. A question was put to a panel of AI models. Your job is NOT \
to repeat them but to synthesize their answers into something useful.

Be concise and concrete. Structure your reply:

1. **Agreement** — where the panel converges (1-3 points).
2. **Disagreement** — where models contradict each other, and who claims what.
3. **Verdict** — your recommendation with reasoning. If information is missing, \
say exactly what's missing instead of padding.

If one of the panel answers is your own earlier answer, judge it as critically \
as the rest.

=== QUESTION ===
{question}

=== PANEL ANSWERS ===
{answers}
"""


def run_judge(judge: Model, question: str, panel: list[Model]) -> str:
    blocks = [f"--- {m.name} ({m.model}) ---\n{m.answer}"
              for m in panel if not m.error]
    if not blocks:
        return "(no successful panel answers — nothing to judge)"
    prompt = JUDGE_PROMPT.format(question=question, answers="\n\n".join(blocks))
    try:
        return call_model(judge, prompt)
    except Exception as e:  # noqa: BLE001
        return f"(judge failed: {str(e)[:300]})"


# --- interactive checkbox picker -------------------------------------------

def pick(items: list[str], preselected: list[bool], title: str) -> list[bool]:
    """Return a bool mask. curses checkboxes, numbered fallback otherwise."""
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return preselected
    try:
        return _pick_curses(items, preselected, title)
    except Exception:  # noqa: BLE001 — curses unavailable/odd terminal
        return _pick_numbered(items, preselected, title)


def _pick_curses(items, preselected, title):
    import curses

    def run(stdscr):
        curses.curs_set(0)
        sel = list(preselected)
        pos = 0
        while True:
            stdscr.erase()
            stdscr.addstr(0, 0, title)
            stdscr.addstr(1, 0, "↑/↓ move · space toggle · a all · n none · enter ok · q cancel")
            for i, it in enumerate(items):
                mark = "[x]" if sel[i] else "[ ]"
                line = f" {mark} {it}"
                if i == pos:
                    stdscr.addstr(3 + i, 0, line, curses.A_REVERSE)
                else:
                    stdscr.addstr(3 + i, 0, line)
            stdscr.refresh()
            k = stdscr.getch()
            if k in (curses.KEY_UP, ord("k")):
                pos = (pos - 1) % len(items)
            elif k in (curses.KEY_DOWN, ord("j")):
                pos = (pos + 1) % len(items)
            elif k == ord(" "):
                sel[pos] = not sel[pos]
            elif k == ord("a"):
                sel = [True] * len(items)
            elif k == ord("n"):
                sel = [False] * len(items)
            elif k in (curses.KEY_ENTER, 10, 13):
                return sel
            elif k in (ord("q"), 27):
                return preselected
    return curses.wrapper(run)


def _pick_numbered(items, preselected, title):
    sel = list(preselected)
    while True:
        print(f"\n{title}")
        for i, it in enumerate(items):
            print(f"  {i + 1}. [{'x' if sel[i] else ' '}] {it}")
        raw = input("Toggle numbers (comma-sep), 'a' all, 'n' none, Enter to confirm: ").strip()
        if raw == "":
            return sel
        if raw.lower() == "a":
            sel = [True] * len(items); continue
        if raw.lower() == "n":
            sel = [False] * len(items); continue
        for part in raw.replace(",", " ").split():
            if part.isdigit() and 1 <= int(part) <= len(items):
                sel[int(part) - 1] = not sel[int(part) - 1]


# --- interactive setup wizard ----------------------------------------------

PROVIDERS_META = {
    "anthropic": {"label": "Anthropic (Claude)", "cli": "claude",
                  "default_model": "claude-opus-4-8", "default_name": "claude"},
    "openai":    {"label": "OpenAI (GPT)", "cli": "codex",
                  "default_model": "gpt-5.5", "default_name": "gpt"},
    "gemini":    {"label": "Google (Gemini)", "cli": None,
                  "default_model": "gemini-2.5-flash", "default_name": "gemini"},
    "deepseek":  {"label": "DeepSeek", "cli": None,
                  "default_model": "deepseek-v4-pro", "default_name": "deepseek"},
}


CLI_INFO = {
    "anthropic": {"bin": "claude", "creds": "~/.claude/.credentials.json",
                  "login": "run `claude`, then /login"},
    "openai":    {"bin": "codex", "creds": "~/.codex/auth.json",
                  "login": "run `codex login`"},
}


def _check_cli(provider: str) -> None:
    """Tell the user if the chosen CLI is missing or not logged in."""
    info = CLI_INFO.get(provider)
    if not info:
        return
    binp = _find_bin(info["bin"])
    installed = _bin_ok(binp)
    logged_in = Path(os.path.expanduser(info["creds"])).exists()
    if installed and logged_in:
        print(f"{GRN}  ✓ {info['bin']} is installed and logged in.{RST}")
    elif not installed:
        print(f"{YEL}  ! {info['bin']} CLI not found. Install it, then {info['login']} "
              f"— required before quorum can use this model.{RST}")
    else:
        print(f"{YEL}  ! {info['bin']} is installed but not logged in. "
              f"{info['login'].capitalize()} before using quorum.{RST}")


def _ask(prompt: str, default: str = "") -> str:
    s = input(f"{prompt}{f' [{default}]' if default else ''}: ").strip()
    return s or default


def _yesno(prompt: str, default: bool = True) -> bool:
    s = input(f"{prompt} [{'Y/n' if default else 'y/N'}]: ").strip().lower()
    return default if not s else s.startswith("y")


def _toml_str(v: str) -> str:
    return json.dumps(v)  # JSON string == valid TOML basic string for plain text


def run_setup() -> None:
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        sys.exit("setup needs an interactive terminal.")
    print(f"{BOLD}quorum setup{RST} — let's build your config.\n")
    if CONFIG_PATH.exists() and not _yesno(
            f"Config already exists at {CONFIG_PATH}. Overwrite?", False):
        print("Aborted."); return

    entries = []
    for prov, meta in PROVIDERS_META.items():
        print(f"\n{BOLD}{CYA}{meta['label']}{RST}")
        if not _yesno("  Include it on the panel?", True):
            continue
        auth = "api"
        if meta["cli"]:
            print(f"  Auth: [1] API key   [2] Local CLI login "
                  f"({meta['cli']} — uses your subscription/login)")
            auth = "cli" if _ask("  Choose 1 or 2", "1") == "2" else "api"
            if auth == "cli":
                _check_cli(prov)
        name = meta["default_name"]  # auto — one provider, one fixed label
        model = _ask("  Model id", meta["default_model"])
        e = {"name": name, "provider": prov, "model": model, "auth": auth}
        if auth == "api":
            key = getpass.getpass(
                "  Paste API key (hidden; blank = read from env later): ").strip()
            e["api_key"] = key or f"env:{prov.upper()}_API_KEY"
        entries.append(e)

    if not entries:
        sys.exit("No providers selected — nothing to write.")

    print(f"\n{BOLD}Judge{RST} — who synthesizes the panel?")
    names = [e["name"] for e in entries]
    for i, n in enumerate(names, 1):
        print(f"  {i}. {n}")
    print("  0. no judge (raw answers only)")
    choice = _ask("  Choose by number", "1")
    judge_use = names[int(choice) - 1] if (
        choice.isdigit() and 1 <= int(choice) <= len(names)) else None

    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# quorum config — generated by `quorum --setup`.",
             "# Keys live here; this file is chmod 600. Keep it private.", ""]
    for e in entries:
        lines += ["[[panel]]",
                  f"name = {_toml_str(e['name'])}",
                  f"provider = {_toml_str(e['provider'])}",
                  f"model = {_toml_str(e['model'])}",
                  f"auth = {_toml_str(e['auth'])}"]
        if "api_key" in e:
            lines.append(f"api_key = {_toml_str(e['api_key'])}")
        lines += ["enabled = true", ""]
    lines.append("[judge]")
    if judge_use:
        lines += ["enabled = true", f"use = {_toml_str(judge_use)}"]
    else:
        lines.append("enabled = false")
    CONFIG_PATH.write_text("\n".join(lines) + "\n")
    os.chmod(CONFIG_PATH, 0o600)
    print(f"\n{GRN}Wrote {CONFIG_PATH} (chmod 600).{RST}")

    if _yesno("Install a `quorum` command in ~/.local/bin?", True):
        binp = Path.home() / ".local" / "bin"
        binp.mkdir(parents=True, exist_ok=True)
        w = binp / "quorum"
        w.write_text(f'#!/usr/bin/env bash\n'
                     f'exec python3 "{Path(__file__).resolve()}" "$@"\n')
        os.chmod(w, 0o755)
        print(f"{GRN}Installed {w}.{RST}")
        if str(binp) not in os.environ.get("PATH", "").split(":"):
            print(f"{YEL}Note: {binp} is not in your PATH. Add to ~/.bashrc:\n"
                  f'  export PATH="$HOME/.local/bin:$PATH"{RST}')
    print(f"\n{BOLD}Done.{RST} Try:  quorum \"your question\"")


# --- web UI launcher (optional, needs requirements-web.txt) -----------------

def run_web(port: int) -> None:
    if not CONFIG_PATH.exists():
        sys.exit(f"{RED}No config:{RST} {CONFIG_PATH}. Run `quorum --setup` first.")
    try:
        import uvicorn
        import webapp
    except ImportError as e:
        sys.exit(f"{RED}Web UI needs extra packages.{RST} Install once:\n"
                 f"  pip install -r requirements-web.txt\n  (missing: {e.name})")
    import threading
    import webbrowser
    url = f"http://127.0.0.1:{port}"
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    print(f"{BOLD}quorum web UI{RST} → {url}   (Ctrl-C to stop)")
    uvicorn.run(webapp.app, host="127.0.0.1", port=port, log_level="warning")


# --- main -------------------------------------------------------------------

def read_question(args) -> str:
    parts = []
    if args.question:
        parts.append(" ".join(args.question))
    if args.file:
        parts.append(Path(args.file).read_text())
    if args.context:
        parts.append(f"\n\n--- CONTEXT ({args.context}) ---\n"
                     + Path(args.context).read_text())
    if not parts and not sys.stdin.isatty():
        parts.append(sys.stdin.read())
    q = "\n".join(parts).strip()
    if not q:
        sys.exit(f"{RED}Empty question.{RST} Pass text, -f FILE, or pipe via stdin.")
    return q


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="quorum",
        description="Ask a panel of LLMs and get a judge's synthesis.")
    ap.add_argument("question", nargs="*", help="the question")
    ap.add_argument("-f", "--file", help="read the question from a file")
    ap.add_argument("-c", "--context", help="attach a file as context")
    ap.add_argument("--panel", help="comma-separated panelist names (skip the picker)")
    ap.add_argument("--judge", help="judge by panelist name (overrides config)")
    ap.add_argument("--no-judge", action="store_true", help="skip synthesis")
    ap.add_argument("--pick", action="store_true",
                    help="force the interactive picker even with --panel")
    ap.add_argument("--save", help="write the transcript to PATH (.md)")
    ap.add_argument("--list", action="store_true", help="list configured models and exit")
    ap.add_argument("--setup", action="store_true",
                    help="interactive wizard: build the config and install the command")
    ap.add_argument("--web", action="store_true",
                    help="launch the local web UI in your browser (needs requirements-web.txt)")
    ap.add_argument("--port", type=int, default=8765, help="web UI port (default 8765)")
    args = ap.parse_args()

    if args.setup or args.question == ["setup"]:
        run_setup()
        return

    if args.web:
        run_web(args.port)
        return

    if not CONFIG_PATH.exists():
        if sys.stdin.isatty() and sys.stdout.isatty():
            print(f"No config at {CONFIG_PATH}.")
            if _yesno("Run the setup wizard now?", True):
                run_setup()
            return
        sys.exit(f"{RED}No config:{RST} {CONFIG_PATH}. Run `quorum --setup`.")

    cfg = load_config()

    if args.list:
        for m in cfg.panel:
            flag = "" if m.enabled else " (disabled)"
            print(f"  {m.name:14} {m.provider}/{m.model}  [{m.auth}]{flag}")
        if cfg.judge:
            print(f"  judge → {cfg.judge.name} ({cfg.judge.provider}/{cfg.judge.model})")
        return

    question = read_question(args)

    # choose the panel
    available = [m for m in cfg.panel if m.enabled]
    if not available:
        sys.exit(f"{RED}No enabled models in config.{RST}")
    if args.panel:
        want = {n.strip() for n in args.panel.split(",")}
        panel = [m for m in available if m.name in want]
    elif sys.stdin.isatty() and sys.stdout.isatty():
        labels = [f"{m.name}  ({m.provider}/{m.model}, {m.auth})" for m in available]
        mask = pick(labels, [True] * len(available), "Select panel models:")
        panel = [m for m, on in zip(available, mask) if on]
    else:
        panel = available
    if args.pick and not args.panel:
        pass  # picker already ran above
    if not panel:
        sys.exit(f"{RED}No panelists selected.{RST}")

    # judge
    judge = cfg.judge
    if args.judge:
        judge = next((m for m in cfg.panel if m.name == args.judge), None)
        if judge is None:
            sys.exit(f"{RED}--judge '{args.judge}' matches no configured model.{RST}")
    use_judge = cfg.judge_enabled and not args.no_judge and judge is not None

    print(f"{BOLD}Question:{RST} {question[:200]}{'…' if len(question) > 200 else ''}")
    print(f"{DIM}Panel: {', '.join(m.name for m in panel)}"
          f"{f' · judge: {judge.name}' if use_judge else ''}{RST}\n")

    with ThreadPoolExecutor(max_workers=len(panel)) as ex:
        futs = {ex.submit(run_model, m, question): m for m in panel}
        for fut in as_completed(futs):
            m = fut.result()
            mark = f"{RED}✗{RST}" if m.error else f"{GRN}✓{RST}"
            print(f"{mark} {m.name} ({m.seconds:.1f}s)")

    print()
    for m in panel:
        print(f"{BOLD}{CYA}━━━ {m.name} ({m.provider}/{m.model}) ━━━{RST}")
        print(f"{RED}ERROR: {m.error}{RST}\n" if m.error else m.answer + "\n")

    judge_text = ""
    if use_judge:
        print(f"{BOLD}{YEL}━━━ JUDGE ({judge.name}: {judge.provider}/{judge.model}) ━━━{RST}")
        judge_text = run_judge(judge, question, panel)
        print(judge_text + "\n")

    if args.save:
        md = [f"# quorum\n", f"## Question\n\n{question}\n"]
        for m in panel:
            md.append(f"## {m.name} ({m.provider}/{m.model}) — {m.seconds:.1f}s\n")
            md.append(f"```\nERROR: {m.error}\n```\n" if m.error else m.answer + "\n")
        if judge_text:
            md.append(f"## Judge ({judge.name})\n\n{judge_text}\n")
        Path(os.path.expanduser(args.save)).write_text("\n".join(md))
        print(f"{DIM}Saved: {args.save}{RST}")

    # exit code: 0 all ok, 1 some failed, 2 all failed
    failed = [m for m in panel if m.error]
    ok = [m for m in panel if not m.error]
    print()
    if not failed:
        print(f"{GRN}All answered: {len(ok)}/{len(panel)}.{RST}")
        sys.exit(0)
    why = lambda m: " ".join(m.error.split())[:140]
    if not ok:
        print(f"{RED}{BOLD}ALL models failed ({len(failed)}/{len(panel)}) — result unreliable.{RST}")
        for m in failed:
            print(f"{RED}  ✗ {m.name}: {why(m)}{RST}")
        sys.exit(2)
    print(f"{YEL}{BOLD}Failed {len(failed)}/{len(panel)}: "
          f"{', '.join(m.name for m in failed)} (judged without them).{RST}")
    for m in failed:
        print(f"{YEL}  ✗ {m.name}: {why(m)}{RST}")
    sys.exit(1)


if __name__ == "__main__":
    main()
