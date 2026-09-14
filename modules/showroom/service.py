"""Independent, resumable read-only ingestion and local-calendar daily scheduling."""

from datetime import datetime, timedelta
import threading
import time
from urllib.error import HTTPError, URLError
from zoneinfo import ZoneInfo

from modules.showroom.reports import generate


class Service:
    def __init__(self, repo, client):
        self.repo, self.client = repo, client
        self.stop = threading.Event()
        self.last_poll = None
        self.last_reports = None
        self.error = None
        self.thread = threading.Thread(
            target=self.run, name="showroom-consumer", daemon=True
        )

    def poll(self):
        for binding in self.repo.bindings():
            if self.stop.is_set():
                return
            if binding["paused"]:
                continue
            cursor = binding["cursor"]
            try:
                # Fairness: a backlogged run cannot monopolize all other cameras/stores.
                for _ in range(4):
                    if self.stop.is_set():
                        return
                    page = self.client.events(binding["run"], cursor)
                    if not self.repo.ingest(
                        binding["store"], binding["run"], cursor, page
                    ):
                        break
                    if page.get("caught_up") or page["cursor"] == cursor:
                        break
                    cursor = page["cursor"]
            except HTTPError as exc:
                self.repo.mark_error(
                    binding["store"],
                    binding["run"],
                    f"追踪接口返回 HTTP {exc.code}；保留游标，等待恢复。",
                )
            except (URLError, OSError, TimeoutError):
                self.repo.mark_error(
                    binding["store"],
                    binding["run"],
                    "追踪服务暂不可用；保留游标，等待恢复。",
                )
            except (ValueError, KeyError, TypeError) as exc:
                message = (
                    str(exc) if isinstance(exc, ValueError) else "追踪事件结构不完整。"
                )
                self.repo.mark_error(binding["store"], binding["run"], message[:240])
        self.last_poll = time.time()

    def daily(self, now=None):
        now = time.time() if now is None else now
        for store in self.repo.stores():
            local = datetime.fromtimestamp(now, ZoneInfo(store["timezone"]))
            # Yesterday is eligible at 00:10. Older days remain eligible before then.
            last = local.date() - timedelta(
                days=1 if (local.hour, local.minute) >= (0, 10) else 2
            )
            first = max(
                datetime.fromtimestamp(
                    store["created_at"], ZoneInfo(store["timezone"])
                ).date(),
                local.date() - timedelta(days=30),
            )
            while first <= last:
                if self.stop.is_set():
                    return
                generate(self.repo, store["id"], first.isoformat(), now)
                first += timedelta(days=1)
        self.last_reports = now

    def run(self):
        next_report = 0
        while not self.stop.is_set():
            try:
                self.poll()
                if time.monotonic() >= next_report:
                    self.daily()
                    next_report = time.monotonic() + 60
                self.error = None
            except Exception as exc:
                # A failure must be visible and retryable, never kill tracking or skip a cursor.
                self.error = type(exc).__name__
            self.stop.wait(1)

    def start(self):
        self.thread.start()

    def close(self):
        self.stop.set()
        if self.thread.ident is not None:
            self.thread.join()  # network calls are individually bounded by the client timeout
