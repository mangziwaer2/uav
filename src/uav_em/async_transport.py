"""One-worker nonblocking callback transport; dispatch is not model-call proof."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import threading
import time


class ThreadAdvisor:
    """Background IO callback receives public data only, never a world/config.

    Callback must have its own finite network timeout. Python cannot forcibly
    kill a running thread; expiry preserves busy() until it actually completes.
    No actual LLM call is inferred from a future's existence or completion.
    """
    def __init__(self, callback, clock=time.monotonic):
        self.callback, self.clock = callback, clock
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='uav-advisor')
        self.current = None
        self.count = 0
        self.closed = False
        self.lock = threading.Lock()

    def busy(self):
        return self.current is not None

    def submit(self, public, now):
        if self.closed or self.busy():
            raise ValueError('advisor is closed or still busy')
        token = f'callback_{self.count:03d}'
        self.count += 1
        record = dict(token=token, issued_physical_seconds=now, issued_wall_seconds=self.clock(),
                      started_wall_seconds=None, received_wall_seconds=None,
                      actual_model_call=None, origin='callback_without_model_receipt')
        payload = deepcopy(public)

        def work():
            with self.lock:
                record['started_wall_seconds'] = self.clock()
            try:
                raw, error = self.callback(payload), None
            except Exception as exc:
                raw, error = None, repr(exc)
            with self.lock:
                record.update(raw=deepcopy(raw), error=error, received_wall_seconds=self.clock())

        future = self.executor.submit(work)
        self.current = dict(token=token, future=future, record=record)
        return token

    def poll(self, now):
        if self.current is None or not self.current['future'].done():
            return []
        item = self.current
        self.current = None
        if item['future'].cancelled():
            return [dict(token=item['token'], raw=None, error='cancelled_before_worker_start',
                         actual_model_call=False, origin='cancelled_callback')]
        item['future'].result()
        with self.lock:
            reply = deepcopy(item['record'])
        reply['visible_at_physical_seconds'] = now
        return [reply]

    def cancel(self, token):
        if self.current is None or self.current['token'] != token:
            return False
        cancelled = self.current['future'].cancel()
        if cancelled:
            self.current = None
        return cancelled

    def close(self):
        self.closed = True
        if self.current:
            self.cancel(self.current['token'])
        self.executor.shutdown(wait=False, cancel_futures=True)
