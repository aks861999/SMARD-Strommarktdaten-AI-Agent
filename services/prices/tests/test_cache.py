import threading
import time
from datetime import date, datetime, timedelta, timezone

import pytest

from prices_service.cache import CachedPriceSource
from prices_service.sources import BERLIN, HourlyPrice, day_hours


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def full_day(target_date, hours=None):
    """Every hour of the Berlin day (23, 24 or 25 of them), or just the first ``hours`` of them."""
    start = datetime.combine(target_date, datetime.min.time(), tzinfo=BERLIN).astimezone(timezone.utc)
    count = day_hours(target_date) if hours is None else hours
    return [HourlyPrice((start + timedelta(hours=i)).astimezone(BERLIN), 50.0 + i) for i in range(count)]


class FakeSource:
    def __init__(self, hours=None, fail=False):
        self.calls, self.hours, self.fail = 0, hours, fail

    def get_prices(self, target_date, region="DE-LU"):
        self.calls += 1
        if self.fail:
            raise RuntimeError("upstream down")
        return full_day(target_date, self.hours)


D = date(2026, 11, 16)


def make(source, **kw):
    clock = Clock()
    return CachedPriceSource(source, clock=clock, **kw), clock


def test_a_complete_day_is_fetched_once_and_then_served_from_cache():
    source = FakeSource()
    cache, clock = make(source)
    assert cache.lookup(D)[1] == "MISS"
    clock.now += 3600                                        # an hour later it is still fresh
    prices, status = cache.lookup(D)
    assert status == "HIT" and len(prices) == 24 and source.calls == 1


def test_a_complete_day_expires_after_the_long_ttl():
    source = FakeSource()
    cache, clock = make(source, complete_ttl=6 * 3600)
    cache.lookup(D)
    clock.now += 6 * 3600 + 1
    assert cache.lookup(D)[1] == "MISS" and source.calls == 2


def test_a_partial_day_is_only_cached_briefly_so_new_prices_show_up():
    source = FakeSource(hours=10)                            # e.g. tomorrow, before the prices are published
    cache, clock = make(source, incomplete_ttl=120)
    cache.lookup(D)
    clock.now += 60
    assert cache.lookup(D)[1] == "HIT"
    clock.now += 61                                          # past 2 minutes: ask again
    assert cache.lookup(D)[1] == "MISS" and source.calls == 2


def test_an_empty_day_is_not_remembered_for_long_either():
    source = FakeSource(hours=0)
    cache, clock = make(source, incomplete_ttl=120)
    assert cache.lookup(D) == ([], "MISS")
    clock.now += 121
    cache.lookup(D)
    assert source.calls == 2


def test_failures_are_never_cached():
    source = FakeSource(fail=True)
    cache, _ = make(source)
    for _ in range(3):
        with pytest.raises(RuntimeError):
            cache.lookup(D)
    assert source.calls == 3 and len(cache) == 0
    source.fail = False
    assert cache.lookup(D)[1] == "MISS"                      # recovers as soon as the source does


@pytest.mark.parametrize("day,expected", [(date(2026, 3, 29), 23), (date(2026, 10, 25), 25), (D, 24)])
def test_clock_change_days_count_as_complete_with_23_or_25_hours(day, expected):
    source = FakeSource()
    cache, _ = make(source)
    assert len(cache.lookup(day)[0]) == expected
    assert cache.lookup(day)[1] == "HIT"                     # complete, so cached (not mistaken for partial)


def test_dates_and_regions_are_cached_separately():
    source = FakeSource()
    cache, _ = make(source)
    cache.lookup(D); cache.lookup(D + timedelta(days=1)); cache.lookup(D, "OTHER")
    assert source.calls == 3 and len(cache) == 3


def test_callers_cannot_corrupt_the_cache_by_mutating_a_result():
    cache, _ = make(FakeSource())
    first = cache.lookup(D)[0]
    first.clear()
    assert len(cache.lookup(D)[0]) == 24


def test_memory_is_bounded():
    cache, clock = make(FakeSource(), max_entries=5)
    for i in range(40):
        cache.lookup(D + timedelta(days=i))
        clock.now += 1
    assert len(cache) <= 5
    assert cache.lookup(D + timedelta(days=39))[1] == "HIT"  # the newest entries are the ones kept


def test_concurrent_requests_for_the_same_day_share_one_fetch():
    class Slow(FakeSource):
        def get_prices(self, target_date, region="DE-LU"):
            time.sleep(0.2)
            return super().get_prices(target_date, region)

    source = Slow()
    cache = CachedPriceSource(source)
    results = []
    threads = [threading.Thread(target=lambda: results.append(cache.lookup(D)[1])) for _ in range(8)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert source.calls == 1
    assert sorted(results) == ["HIT"] * 7 + ["MISS"]
