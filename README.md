# quorum

Ask one question to a **panel of LLMs** at once, then let a **judge** model
synthesize their answers into agreement / disagreement / a verdict.

You choose who sits on the panel and who judges — with checkboxes or flags.
Each model can be reached through its **official API** (your key) or through
your **local CLI login** (`claude -p` / `codex exec`) — your choice, per model.

- **Zero dependencies.** Pure Python standard library. No `pip install`.
- **Pick your panel.** Interactive checkboxes, or `--panel a,b,c` for scripts.
- **Pick your judge.** Any model, or none.
- **Providers:** Anthropic (Claude), OpenAI (GPT), Google (Gemini), DeepSeek.

## Why

A single model can be confidently wrong. Asking several independent models the
same question — and having one reconcile them — surfaces disagreement you'd
otherwise never see. quorum makes that a one-liner.

## Install

```bash
git clone <your-repo-url> quorum
cd quorum
cp config.example.toml ~/.config/quorum/config.toml   # mkdir -p ~/.config/quorum first
# edit the config, set your keys (or use auth = "cli")
python3 quorum.py "your question"
```

Optional convenience wrapper:

```bash
printf '#!/usr/bin/env bash\nexec python3 %s/quorum.py "$@"\n' "$PWD" > ~/.local/bin/quorum
chmod +x ~/.local/bin/quorum
```

Requirements: Python 3.11+ (for the built-in `tomllib`). For `auth = "cli"`,
the relevant CLI must be installed and logged in (`claude` / `codex`).

## Auth: API vs CLI

| Mode | How it reaches the model | Cost | Notes |
|------|--------------------------|------|-------|
| `auth = "api"` | Official API with your key | Pay per token | Portable, reproducible. Recommended. |
| `auth = "cli"` | Your local CLI login (`claude -p`, `codex exec`) | Your subscription/plan | Anthropic & OpenAI only. **Driving a subscription CLI programmatically may conflict with the provider's terms — that's on you to verify.** |

API keys are read from the environment via `api_key = "env:VAR"` (recommended)
so secrets never live in the config file. Gemini and DeepSeek are API-only.

## Usage

```bash
quorum "Is now a good time to rotate from alts into BTC?"   # interactive picker
quorum --panel gpt,gemini,claude "..."                      # explicit panel
quorum --judge claude --panel gpt,deepseek "..."            # choose the judge
quorum --no-judge "..."                                     # raw answers only
quorum -c strategy.py "find the weak spots"                 # attach a file
cat log.txt | quorum "what's wrong here?"                   # question via stdin
quorum --list                                               # show configured models
quorum --save out.md "..."                                  # write a transcript
```

The picker: `↑/↓` move, `space` toggle, `a` all, `n` none, `enter` confirm,
`q` cancel. If your terminal can't run curses (or output is piped), it falls
back to a numbered prompt automatically.

## Exit codes

`0` all models answered · `1` some failed (judged without them; names + reasons
printed) · `2` all failed (result unreliable). Handy for scripts and cron.

## Adding a provider

Each provider is a small function in `quorum.py` (`_api_*` / `_cli_*`) wired in
`call_model()`. Add one function, add one branch. No framework.

## License

MIT — see LICENSE.
