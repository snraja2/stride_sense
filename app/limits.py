"""Small in-memory live-run rate limiter for hackathon deployments."""

from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timezone
from threading import Lock
import time


class RunLimiter:
    def __init__(self, per_ip_per_hour: int = 3, daily_cap: int = 60) -> None:
        self.per_ip_per_hour = max(1, int(per_ip_per_hour))
        self.daily_cap = max(1, int(daily_cap))
        self._ip_runs: dict[str, deque[float]] = defaultdict(deque)
        self._daily_runs = 0
        self._day = datetime.now(timezone.utc).date()
        self._lock = Lock()

    def allow(self, ip: str, *, live: bool) -> bool:
        """Return true and count the run, or false when live limits are reached."""
        if not live:
            return True
        now = time.monotonic()
        today = datetime.now(timezone.utc).date()
        with self._lock:
            if today != self._day:
                self._day = today
                self._daily_runs = 0
                self._ip_runs.clear()
            recent = self._ip_runs[ip]
            while recent and now - recent[0] >= 3600:
                recent.popleft()
            if len(recent) >= self.per_ip_per_hour or self._daily_runs >= self.daily_cap:
                return False
            recent.append(now)
            self._daily_runs += 1
            return True
