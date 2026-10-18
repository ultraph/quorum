"""Private, atomic connection overrides for the single-user web installation."""
import json
import os
import tempfile
import threading
from pathlib import Path

_LOCK = threading.Lock()


def path_for(config_path):
    return Path(config_path).with_name("connections.local.json")


def read(config_path):
    path = path_for(config_path)
    return json.loads(path.read_text()) if path.exists() else {}


def save_claude(config_path, auth, key=None, remove=False, provider="anthropic"):
    if provider not in ("anthropic", "openai"):
        raise ValueError("Unsupported connection provider")
    with _LOCK:
        data = read(config_path)
        entry = data.setdefault(provider, {})
        entry["auth"] = auth
        if remove:
            entry["api_key"] = ""
        elif key is not None:
            entry["api_key"] = key
        path = path_for(config_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=".connections-", dir=path.parent)
        try:
            with os.fdopen(fd, "w") as stream:
                json.dump(data, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(name, path)
        finally:
            if os.path.exists(name):
                os.unlink(name)
