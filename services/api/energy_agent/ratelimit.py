"""In-memory, per-client rate limiting for the chat API (one process, nothing persisted).

Two small pieces, both free of FastAPI so they are easy to test:

* ``SlidingWindowLimiter``: at most N events per window for each key, for several (N, window) rules at
  once. ``retry_after`` only looks; ``record`` spends budget. A caller can therefore ask several limiters
  and spend from all of them only when every one agrees.
* ``client_ip``: decides which address counts as "the client". Behind a proxy the socket address is the
  proxy's, so a forwarded header is needed, but a client can send any header it likes, so the header is
  used only when the operator names it (``CLIENT_IP_HEADER``) and, for ``X-Forwarded-For``, only the entry
  appended by the Nth trusted proxy from the right is believed (never the leftmost, which is forgeable).
"""
from __future__ import annotations

import ipaddress
import threading
import time
from collections import deque
from typing import Callable, Iterable, Mapping


class SlidingWindowLimiter:
    def __init__(self, rules: Iterable[tuple[int, float]], max_keys: int = 10_000,
                 clock: Callable[[], float] = time.monotonic):
        # A rule is (max events, window in seconds). Non-positive values switch that rule off.
        self.rules = [(int(n), float(w)) for n, w in rules if n > 0 and w > 0]
        self._longest = max((w for _, w in self.rules), default=0.0)
        self._max_keys = max_keys
        self._clock = clock
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return bool(self.rules)

    def retry_after(self, key: str) -> float:
        """Seconds until one more event for ``key`` is allowed; 0.0 if it is allowed now. Changes nothing."""
        if not self.rules:
            return 0.0
        now = self._clock()
        with self._lock:
            hits = self._hits.get(key)
            if not hits:
                return 0.0
            wait = 0.0
            for limit, window in self.rules:
                recent = [t for t in hits if now - t < window]  # timestamps are kept in ascending order
                if len(recent) >= limit:
                    # One slot frees up when the (len - limit + 1)-th oldest recent event leaves the window.
                    wait = max(wait, recent[len(recent) - limit] + window - now)
            return max(0.0, wait)

    def record(self, key: str) -> None:
        """Count one event for ``key`` now."""
        if not self.rules:
            return
        now = self._clock()
        with self._lock:
            hits = self._hits.setdefault(key, deque())
            while hits and now - hits[0] >= self._longest:
                hits.popleft()
            hits.append(now)
            if len(self._hits) > self._max_keys:
                self._collect_garbage(now)

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()

    def _collect_garbage(self, now: float) -> None:
        """Bound memory: drop keys that have gone quiet, then the least recently active ones."""
        for key in [k for k, h in self._hits.items() if not h or now - h[-1] >= self._longest]:
            del self._hits[key]
        excess = len(self._hits) - self._max_keys
        if excess > 0:
            for key in sorted(self._hits, key=lambda k: self._hits[k][-1])[:excess]:
                del self._hits[key]


def normalize_ip(value: str) -> str | None:
    """Canonical key for an IP literal, or None if it is not one.

    IPv6 clients are grouped by their /64 so that rotating through addresses of one network does not
    hand out a fresh allowance each time. IPv4-mapped IPv6 addresses count as the IPv4 address.
    """
    try:
        ip = ipaddress.ip_address(value.strip())
    except ValueError:
        return None
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped is not None:
            return str(ip.ipv4_mapped)
        return f"{ipaddress.ip_network(f'{ip}/64', strict=False).network_address}/64"
    return str(ip)


def client_ip(headers: Mapping[str, str], peer: str | None, header: str = "", hops: int = 1) -> tuple[str, str]:
    """Return ``(key, source)``: the address to rate-limit and where it came from.

    ``source`` is ``"peer"`` (the socket address) or the lower-case header name that was used. Any missing,
    short or malformed header value falls back to the peer, so a bad header can never crash a request.
    """
    fallback = normalize_ip(peer or "") or (peer or "unknown")
    name = (header or "").strip().lower()
    if not name:
        return fallback, "peer"
    raw = headers.get(name)
    if not raw:
        return fallback, "peer"
    if name == "x-forwarded-for":
        # "client, proxy1, proxy2": every trusted proxy appends the address it saw, so with N trusted
        # proxies the real client is the Nth entry from the right. Entries further left are forgeable.
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        n = max(1, int(hops))
        if len(parts) < n:
            return fallback, "peer"
        candidate = parts[-n]
    else:
        candidate = raw.split(",")[0].strip()
    ip = normalize_ip(candidate)
    return (ip, name) if ip else (fallback, "peer")
