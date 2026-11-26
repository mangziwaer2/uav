"""Isolated stdin transport for one public route tool request per session."""

import json
import math
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from time import monotonic

from .passive_repair_checkpoint import ISOLATED_FLAGS, ISOLATION, _json, _profile_code, _write, parse_session
from .passive_route_exchange import ReceivedReply, _sha256


_INHERITED = {"PATH", "LANG", "LC_ALL", "TZ", "SSL_CERT_FILE", "SSL_CERT_DIR", "HTTPS_PROXY", "NO_PROXY"}


def isolated_stdio_transport(out, executable, profile, *, timeout_s, total_timeout_s=None):
    """Keep raw public input and process evidence; do not retry a failed round."""
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("positive receiver timeout required")
    if total_timeout_s is not None and (not math.isfinite(total_timeout_s) or total_timeout_s <= 0):
        raise ValueError("positive total receiver timeout required")
    executable = executable.resolve(strict=True)
    config = _profile_code(profile)
    token = profile[1].get("experimental_bearer_token", "").encode()
    command = [str(executable), "exec", "--json", "--ephemeral", "--skip-git-repo-check",
               "--ignore-rules", "--sandbox", "read-only", "--color", "never", *ISOLATED_FLAGS,
               "-c", "approval_policy=\"never\"", "-c", f"developer_instructions={json.dumps(ISOLATION)}", "-"]
    env = {key: value for key, value in os.environ.items() if key in _INHERITED}
    out.mkdir()
    round_index = 0
    model_elapsed = 0.0

    def transport(wire, expected_sha256):
        nonlocal round_index, model_elapsed
        round_index += 1
        prefix = f"{round_index:04d}"
        actual_sha256 = _sha256(wire)
        _write(out / f"{prefix}-request.json", wire)
        result = dict(wire_bytes=len(wire), expected_sha256=expected_sha256, received_sha256=actual_sha256,
                      status="delivery_mismatch", session_id=None)
        events, diagnostics, raw = b"", b"", b""
        remaining = timeout_s if total_timeout_s is None else min(timeout_s, total_timeout_s - model_elapsed)
        if actual_sha256 == expected_sha256 and remaining <= 0:
            result["status"] = "total_timeout"
        elif actual_sha256 == expected_sha256:
            started = monotonic()
            try:
                with TemporaryDirectory(prefix="uav-stdio-") as directory:
                    root = Path(directory)
                    home, work = root / "home", root / "empty"
                    home.mkdir(mode=0o700)
                    work.mkdir(mode=0o700)
                    descriptor = os.open(home / "config.toml", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                    with os.fdopen(descriptor, "wb") as stream:
                        stream.write(config)
                    child_env = dict(env, HOME=str(home), CODEX_HOME=str(home))
                    process = subprocess.run(command, cwd=work, env=child_env, input=wire, capture_output=True,
                                             timeout=remaining, check=False)
                    events, diagnostics = process.stdout, process.stderr
                    result.update(status="process_failed", exit_code=process.returncode)
                    if process.returncode == 0:
                        result["status"] = "invalid_session_events"
                        session_id, raw = parse_session(events)
                        result.update(status="delivered", session_id=session_id)
            except subprocess.TimeoutExpired as error:
                events, diagnostics = error.stdout or b"", error.stderr or b""
                result["status"] = "timeout"
            except OSError as error:
                result.update(status="process_error", errno=error.errno)
            except (ValueError, TypeError, KeyError, AttributeError, RecursionError):
                result["status"] = "invalid_session_events"
            finally:
                model_elapsed += monotonic() - started
        if token and (token in events or token in diagnostics):
            events, diagnostics = (body.replace(token, b"[redacted]") for body in (events, diagnostics))
            result.update(status="credential_redacted", session_id=None)
            raw = b""
        for suffix, body in (("session.jsonl", events), ("stderr.txt", diagnostics), ("reply.txt", raw),
                             ("receiver.json", _json(result))):
            _write(out / f"{prefix}-{suffix}", body)
        if result["status"] not in ("delivered", "delivery_mismatch"):
            raise OSError(f"isolated receiver: {result['status']}")
        return ReceivedReply(actual_sha256, result["session_id"], raw)

    return transport
