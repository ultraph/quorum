# quorum

Ask one question to a **panel of LLMs** at once, then let a **judge** model
synthesize their answers into agreement / disagreement / a verdict.

You choose who sits on the panel and who judges - with checkboxes or flags.
Each model can be reached through its **official API** (your key) or through
your **local CLI login** (`claude -p` / `codex exec`) - your choice, per model.

- **Zero dependencies.** The CLI is pure Python standard library. No `pip install`.
- **Pick your panel.** Interactive checkboxes, or `--panel a,b,c` for scripts.
- **Pick your judge.** Any model, or none.
- **Providers:** Anthropic (Claude), OpenAI (GPT), Google (Gemini), DeepSeek.
- **Optional web UI.** A local browser front-end (`--web`) with rendered
  Markdown, light/dark themes, pasted outside opinions, and file attachments.

## Why

A single model can be confidently wrong. Asking several independent models the
same question - and having one reconcile them - surfaces disagreement you'd
otherwise never see. quorum makes that a one-liner.

## Install

```bash
git clone https://github.com/ultraph/quorum.git
cd quorum
python3 quorum.py --setup
```

The **interactive setup wizard** walks you through it: pick which providers to
include, choose API key or local CLI per model, paste your keys (entered
hidden), choose a judge, and it writes `~/.config/quorum/config.toml`
(chmod 600) and installs a `quorum` command in `~/.local/bin`. Then:

```bash
quorum "your question"
```

Running `quorum` with no config offers to launch the wizard automatically.

## Requirements

- **Python 3.8+**. No other dependencies. (On 3.11+ the config is parsed by the
  built-in `tomllib`; on older versions like Ubuntu 22.04 / Mint 21's Python 3.10,
  a small bundled stdlib-only TOML reader is used automatically.)
- **`auth = "api"`** - just an API key for that provider. No CLI needed.
- **`auth = "cli"`** - the provider's CLI must be installed **and logged in**:
  - **Anthropic → `claude`** (Claude Code) - log in: run `claude`, then `/login`.
  - **OpenAI → `codex`** (Codex CLI) - log in: `codex login`.

The setup wizard checks the CLI tools are installed and logged in. If a CLI
model errors with the tool printing its own banner/usage, update that CLI
(e.g. `npm install -g @openai/codex@latest`) - very old versions can reject
flags quorum uses.

### Manual setup (optional)

Prefer to edit by hand? Copy the example and fill it in:

```bash
mkdir -p ~/.config/quorum
cp config.example.toml ~/.config/quorum/config.toml
# edit it: set keys (or "env:VAR"), or use auth = "cli"
```

## Auth: API vs CLI

| Mode | How it reaches the model | Cost | Notes |
|------|--------------------------|------|-------|
| `auth = "api"` | Official API with your key | Pay per token | Portable, reproducible. Recommended. |
| `auth = "cli"` | Your local CLI login (`claude -p`, `codex exec`) | Your subscription/plan | Anthropic & OpenAI only. **Driving a subscription CLI programmatically may conflict with the provider's terms - that's on you to verify.** |

API keys are read from the environment via `api_key = "env:VAR"` (recommended)
so secrets never live in the config file. Gemini and DeepSeek are API-only.

## Usage

```bash
quorum "Is now a good time to rotate from alts into BTC?"   # interactive picker
quorum --panel gpt,gemini,claude "..."                      # explicit panel
quorum --judge claude --panel gpt,deepseek "..."            # choose the judge
quorum --no-judge "..."                                     # raw answers only
quorum -c strategy.py "find the weak spots"                 # attach a file as context
cat error.log | quorum                                      # whole pipe becomes the question
quorum --list                                               # show configured models
quorum --save out.md "..."                                  # write a transcript
```

The picker: `↑/↓` move, `space` toggle, `a` all, `n` none, `enter` confirm,
`q` cancel. If your terminal can't run curses (or output is piped), it falls
back to a numbered prompt automatically.

## Web UI (optional)

Prefer a browser over the terminal? quorum ships a local web UI: tick the panel
models, pick a judge, ask, and watch each answer stream in live with the
judge's verdict synthesized on top.

```bash
pip install -r requirements-web.txt   # one-time; the CLI itself stays dependency-free
quorum --web                          # opens http://127.0.0.1:8765 in your browser
quorum --web --port 9000              # custom port
```

What the browser adds over the terminal:

- **Rendered Markdown** — answers and the verdict display as formatted text
  (headings, lists, code blocks), not raw `**`/`#`.
- **Verdict on top, answers collapsed** — the judge's synthesis stays expanded
  at the top; each model's answer is a collapsible card you open on demand.
- **Live feedback** — a skeleton card with a running timer while each model
  thinks, per-provider accent colors, and a final summary pill.
- **Light & dark themes** — toggle in the header; your choice is remembered and
  the system preference is the default.
- **Paste an outside opinion** — drop in an answer from another chat (e.g. a
  model you don't have configured); it joins the panel as a card and the judge
  weighs it alongside the live models.
- **Attach a file** — add a text file (`.md`, `.txt`, code, …) and the whole
  panel and the judge analyze its contents.

It binds to `127.0.0.1` only (localhost) — it runs with your keys, so it is not
exposed to the network. The CLI has zero dependencies; these extras are needed
only for `--web`.

### Hit an "externally-managed environment" error?

On recent Debian / Ubuntu / Linux Mint, `pip install` refuses to touch the
system Python on purpose (to avoid breaking it). That's not a bug. Install the
web extras into an isolated **virtual environment (venv)** instead - no sudo,
nothing touches your system:

```bash
cd ~/quorum                                  # the cloned repo folder
python3 -m venv .venv                         # create an isolated environment
.venv/bin/pip install -r requirements-web.txt # install web deps inside it
.venv/bin/python quorum.py --web              # run the web UI from it
```

What each line does:

1. go into the project folder;
2. create `.venv` - a private Python that lives in the project, separate from
   the system one;
3. install the web dependencies **into that venv** (the system stays untouched);
4. launch the web UI using the venv's Python.

From then on, start the web UI with the same last line:
`.venv/bin/python quorum.py --web`. The plain CLI (`python3 quorum.py "..."`)
still needs no venv - it has zero dependencies.

**If step 2 fails** with a message about `ensurepip` or `python3-venv`, the
venv module isn't installed yet - add it once, then re-run:

```bash
sudo apt install python3-venv
```

## Exit codes

`0` all models answered · `1` some failed (judged without them; names + reasons
printed) · `2` all failed (result unreliable). Handy for scripts and cron.

## Adding a provider

Each provider is a small function in `quorum.py` (`_api_*` / `_cli_*`) wired in
`call_model()`. Add one function, add one branch. No framework.

## License

MIT - see LICENSE.
