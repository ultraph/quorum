"""Pipeline run history: each run is one private JSON file, so results survive
page reloads and service restarts. Single-user installation, no database."""
import json
import os
import tempfile
import time
from pathlib import Path

KEEP = 200  # newest runs kept on disk


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
        out.append({"id": run["id"], "created": run["created"], "question": run["question"][:120],
                    "status": run["status"], "ok": run["ok"], "parent": run.get("parent")})
    return out
