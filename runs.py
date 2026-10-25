"""Pipeline run history: each run is one private JSON file, so results survive
page reloads and service restarts. Single-user installation, no database."""
import json
import os
import re
import tempfile
import time
from pathlib import Path

KEEP = 200  # newest runs kept on disk
_X_POST = re.compile(r"https?://(?:www\.)?(?:x|twitter)\.com/(\w+)\S*")
_URL = re.compile(r"https?://(?:www\.)?([^/\s]+)\S*")


_TOPIC = re.compile(r"^[\s*_#]*Тема:[\s*_]*(.+?)[\s*_.]*$", re.M)


def topic(research: str) -> str:
    """The 2–5 word topic Grok puts on the first line of its research."""
    found = _TOPIC.search(research[:500])
    return found.group(1)[:80] if found else ""


def title(question: str, topic: str = "") -> str:
    """History label: X account name + Grok's short topic. Without a topic,
    links shrink to the account name / domain and the text is cut short."""
    if topic:
        handle = _X_POST.search(question)
        return f"{handle.group(1)} · {topic}" if handle else topic
    text = _URL.sub(r"\1", _X_POST.sub(r"\1 ·", question))
    return " ".join(text.split()).strip(" ·")[:100]


def runs_dir() -> Path:
    path = Path(os.environ.get("QUORUM_RUNS_DIR",
                               Path.home() / ".local" / "state" / "quorum" / "runs"))
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    return path


def new(question: str, research: str, revision: str) -> dict:
    now = time.time()
    run_id = time.strftime("%Y%m%d-%H%M%S", time.localtime(now)) + f"-{int(now * 1000) % 1000:03d}"
    return {"id": run_id, "created": time.strftime("%d.%m.%Y %H:%M", time.localtime(now)),
            "question": question, "research": research, "revision": revision,
            "status": "running", "ok": None, "events": []}


def save(run: dict) -> None:
    directory = runs_dir()
    fd, name = tempfile.mkstemp(prefix=".run-", dir=directory)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(run, stream, ensure_ascii=False)
        os.replace(name, directory / f"{run['id']}.json")
    finally:
        if os.path.exists(name):
            os.unlink(name)


def load(run_id: str) -> dict | None:
    if not run_id.replace("-", "").isdigit():
        return None
    path = runs_dir() / f"{run_id}.json"
    return json.loads(path.read_text()) if path.exists() else None


def recent(limit: int = 30) -> list[dict]:
    files = sorted(runs_dir().glob("*.json"), reverse=True)
    for old in files[KEEP:]:
        old.unlink(missing_ok=True)
    out = []
    for path in files[:limit]:
        try:
            run = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        out.append({"id": run["id"], "created": run["created"], "title": title(run["question"], run.get("topic", "")),
                    "status": run["status"], "ok": run["ok"], "parent": run.get("parent"),
                    "chat": sum(1 for c in run.get("chat", []) if c["role"] == "user")})
    return out


def cost(run: dict) -> dict:
    """Estimated API spend of one run: its own pipeline stages (stages reused
    from a parent carry no cost) and its conversation turns."""
    stages = [e for e in run["events"] if e.get("type") == "stage"]
    chat = [c for c in run.get("chat", []) if c.get("role") == "assistant"]
    return {"pipeline": sum(e.get("cost_usd") or 0 for e in stages),
            "chat": sum(c.get("cost_usd") or 0 for c in chat), "turns": len(chat),
            "est": any(e.get("tokens_est") for e in stages + chat)}


def thread(run_id: str) -> list[dict]:
    """The run's whole topic: the root run and every revision made from it
    (or from its revisions), oldest first."""
    every = {}
    for path in runs_dir().glob("*.json"):
        try:
            item = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        every[item["id"]] = item
    root = run_id
    while every.get(root, {}).get("parent") in every:
        root = every[root]["parent"]
    children: dict[str, list[str]] = {}
    for item in every.values():
        children.setdefault(item.get("parent"), []).append(item["id"])
    out, todo = [], [root]
    while todo:
        current = todo.pop()
        if current in every:
            out.append(every[current])
            todo += children.get(current, [])
    return sorted(out, key=lambda item: item["id"])
