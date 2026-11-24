"""Synchronous, bounded delivery of one public request to a configured worker."""

from dataclasses import dataclass
import hashlib
import math
import os
import selectors
import signal
import subprocess
from tempfile import TemporaryFile
import time

from .passive_model_advice import MAX_INPUT_BYTES, MAX_REPLY_BYTES, MODELS, parse_reply


@dataclass(frozen=True)
class Delivery:
    model: str
    status: str
    request_sha256: str
    input_bytes: int
    raw_reply: bytes
    elapsed_s: float
    worker_started: bool
    return_code: int | None = None


def deliver(command, request, fallback, *, timeout_s):
    """Run one trusted worker command without a shell, network or automatic retry."""
    if (isinstance(command, (str, bytes)) or not command or not isinstance(request, bytes)
            or fallback not in MODELS or not math.isfinite(timeout_s) or timeout_s <= 0):
        raise ValueError("one worker command, byte request, fallback and positive deadline required")
    digest = hashlib.sha256(request).hexdigest()

    def result(status, started=False, raw=b"", elapsed=0.0, code=None, model=fallback):
        return Delivery(model, status, digest, len(request), raw, elapsed, started, code)

    if len(request) > MAX_INPUT_BYTES:
        return result("input_budget")
    with TemporaryFile() as transport:
        transport.write(request)
        transport.seek(0)
        start = time.monotonic()
        deadline = start + timeout_s
        try:
            worker = subprocess.Popen(command, stdin=transport, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                      start_new_session=True)
        except OSError:
            return result("launch_error", elapsed=time.monotonic() - start)
        raw = bytearray()
        status = "timeout"
        code = None
        wire_cap = MAX_REPLY_BYTES + 1
        try:
            with selectors.DefaultSelector() as readiness:
                readiness.register(worker.stdout, selectors.EVENT_READ)
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or not readiness.select(remaining) or time.monotonic() >= deadline:
                        break
                    chunk = os.read(worker.stdout.fileno(), wire_cap + 1 - len(raw))
                    if chunk:
                        raw.extend(chunk)
                        if len(raw) > wire_cap:
                            status = "output_budget"
                            break
                    else:
                        try:
                            code = worker.wait(timeout=max(0, deadline - time.monotonic()))
                        except subprocess.TimeoutExpired:
                            break
                        if time.monotonic() < deadline:
                            status = "received" if code == 0 else "worker_error"
                        break
        finally:
            if code is None:
                try:
                    os.killpg(worker.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                worker.wait()
            worker.stdout.close()

    elapsed = time.monotonic() - start
    if status != "received":
        return result(status, True, bytes(raw), elapsed, worker.returncode)
    try:
        reply = parse_reply(raw.decode("utf-8").removesuffix("\n").removesuffix("\r"))
    except ValueError:
        return result("invalid_reply", True, bytes(raw), elapsed, code)
    return result("accepted", True, bytes(raw), elapsed, code, reply["model"])
