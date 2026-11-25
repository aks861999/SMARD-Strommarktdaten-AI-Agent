"""A thread-safe, bounded TTL cache in front of any price source.

How long an answer may be reused depends on what it is:

* A COMPLETE day (every expected hour present; 23, 24 or 25 around the clock changes) is final,
  because day-ahead results do not change once published. It is kept for ``complete_ttl`` (6 h).
* An EMPTY or PARTIAL day is kept only for ``incomplete_ttl`` (2 min). Tomorrow's prices appear around
  midday; caching "no data" for long would keep reporting nothing after they were published.
* A FAILURE (network error, bad response) is never cached: the exception goes to the caller.

Concurrent requests for the same day share one fetch ("single flight") so a burst of identical questions
cannot hammer the upstream site, and the memory used is capped by ``max_entries``.
"""
from __future__ import annotations

import threading
import time
from datetime import date
from typing import Callable

from .sources import HourlyPrice, PriceSource, day_hours


class CachedPriceSource:
    def __init__(self, inner: PriceSource, *, complete_ttl: float = 6 * 3600, incomplete_ttl: float = 120,
                 max_entries: int = 256, clock: Callable[[], float] = time.monotonic) -> None:
        self.inner = inner
        self.complete_ttl, self.incomplete_ttl = complete_ttl, incomplete_ttl
        self.max_entries = max_entries
        self._clock = clock
        self._entries: dict[tuple[str, date], tuple[float, tuple[HourlyPrice, ...]]] = {}
        self._flight_locks: dict[tuple[str, date], threading.Lock] = {}
        self._lock = threading.Lock()

    def get_prices(self, target_date: date, region: str = "DE-LU") -> list[HourlyPrice]:
        return self.lookup(target_date, region)[0]

    def lookup(self, target_date: date, region: str = "DE-LU") -> tuple[list[HourlyPrice], str]:
        """Return ``(prices, "HIT" | "MISS")``."""
        key = (region, target_date)
        cached = self._fresh(key)
        if cached is not None:
            return list(cached), "HIT"
        with self._flight_lock(key):
            cached = self._fresh(key)  # another request may have filled it while this one waited
            if cached is not None:
                return list(cached), "HIT"
            prices = self.inner.get_prices(target_date, region)  # exceptions propagate; nothing is cached
            ttl = self.complete_ttl if len(prices) == day_hours(target_date) else self.incomplete_ttl
            self._store(key, prices, ttl)
            return list(prices), "MISS"

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._flight_locks.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def _fresh(self, key):
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            if self._clock() >= entry[0]:
                del self._entries[key]
                return None
            return entry[1]

    def _store(self, key, prices: list[HourlyPrice], ttl: float) -> None:
        with self._lock:
            now = self._clock()
            self._entries[key] = (now + ttl, tuple(prices))
            if len(self._entries) > self.max_entries:
                for stale in [k for k, (expires, _) in self._entries.items() if expires <= now]:
                    del self._entries[stale]
                excess = len(self._entries) - self.max_entries
                if excess > 0:  # still too many: drop the entries that would expire soonest
                    for k in sorted(self._entries, key=lambda k: self._entries[k][0])[:excess]:
                        del self._entries[k]

    def _flight_lock(self, key) -> threading.Lock:
        with self._lock:
            if len(self._flight_locks) > 4 * self.max_entries:  # keep this map bounded too
                self._flight_locks = {k: v for k, v in self._flight_locks.items() if k in self._entries or v.locked()}
            return self._flight_locks.setdefault(key, threading.Lock())
