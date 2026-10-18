"""Read CLI account metadata; never return credentials or account identities."""

import json
import os
import selectors
import subprocess
import time
from functools import lru_cache


def _codex_account(binary):
    proc = subprocess.Popen([binary, "app-server", "--listen", "stdio://"],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL)
    try:
        def send(message):
            proc.stdin.write((json.dumps(message) + "\n").encode())
            proc.stdin.flush()

        send({"id": 1, "method": "initialize", "params": {
            "clientInfo": {"name": "quorum", "version": "1.0"}}})
        pending = b""
        deadline = time.monotonic() + 10
        with selectors.DefaultSelector() as selector:
            selector.register(proc.stdout, selectors.EVENT_READ)
            while time.monotonic() < deadline:
                if not selector.select(max(0, deadline - time.monotonic())):
                    break
                chunk = os.read(proc.stdout.fileno(), 65536)
                if not chunk:
                    break
                pending += chunk
                while b"\n" in pending:
                    line, pending = pending.split(b"\n", 1)
                    if not line.strip():
                        continue
                    message = json.loads(line)
                    if message.get("error"):
                        raise ValueError("account protocol error")
                    if message.get("id") == 1:
                        send({"method": "initialized"})
                        send({"id": 2, "method": "account/read",
                              "params": {"refreshToken": False}})
                    elif message.get("id") == 2:
                        return message["result"].get("account")
        raise TimeoutError("account status timeout")
    finally:
        if proc.poll() is None:
            proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        proc.stdin.close()
        proc.stdout.close()


@lru_cache(maxsize=32)
def _cached_status(provider, binary, bucket):
    try:
        if provider == "anthropic":
            result = subprocess.run([binary, "auth", "status", "--json"],
                                    capture_output=True, text=True, timeout=10)
            data = json.loads(result.stdout)
            logged_in = data.get("loggedIn")
            if not isinstance(logged_in, bool):
                raise ValueError("unknown auth status")
            mode = "subscription" if data.get("authMethod") == "claude.ai" else "api"
            plan = data.get("subscriptionType") if logged_in else None
        elif provider == "openai":
            data = _codex_account(binary)
            logged_in = data is not None
            mode = "subscription" if data and data.get("type") in ("chatgpt", "chatgptAuthTokens") else "api"
            plan = data.get("planType") if data else None
        else:
            return {"status": "unknown"}
        # Only permit short plan labels from CLI output. No raw payload leaves
        # this module (payloads can contain email, account IDs, or credentials).
        if not isinstance(plan, str) or len(plan) > 50:
            plan = None
        return {"status": "signed_in" if logged_in else "signed_out",
                "mode": mode, "plan": plan, "checked_at": int(time.time())}
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError):
        return {"status": "unknown", "checked_at": int(time.time())}


def account_status(provider, binary):
    return dict(_cached_status(provider, binary, int(time.time() // 60)))
