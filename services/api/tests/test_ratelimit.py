import pytest

from energy_agent.ratelimit import SlidingWindowLimiter, client_ip, normalize_ip


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def make(rules, **kw):
    clock = Clock()
    return SlidingWindowLimiter(rules, clock=clock, **kw), clock


# ---- the limiter -----------------------------------------------------------------------------

def test_allows_up_to_the_limit_then_reports_how_long_to_wait():
    limiter, clock = make([(3, 60)])
    for _ in range(3):
        assert limiter.retry_after("a") == 0
        limiter.record("a")
        clock.now += 1
    assert limiter.retry_after("a") == pytest.approx(57)   # the first event leaves the window at t=60


def test_budget_comes_back_as_the_window_slides():
    limiter, clock = make([(2, 10)])
    limiter.record("a"); clock.now += 4; limiter.record("a")
    assert limiter.retry_after("a") == pytest.approx(6)
    clock.now += 6.01
    assert limiter.retry_after("a") == 0                    # the older event has expired, one slot is free


def test_retry_after_only_looks_and_record_spends():
    limiter, _ = make([(1, 60)])
    for _ in range(5):
        assert limiter.retry_after("a") == 0                # asking never uses budget
    limiter.record("a")
    assert limiter.retry_after("a") > 0


def test_keys_are_independent():
    limiter, _ = make([(1, 60)])
    limiter.record("a")
    assert limiter.retry_after("a") > 0
    assert limiter.retry_after("b") == 0


def test_the_strictest_rule_decides_when_several_apply():
    limiter, clock = make([(2, 60), (5, 3600)])             # burst of 2 a minute, 5 an hour
    for _ in range(2):
        limiter.record("a")
    assert limiter.retry_after("a") == pytest.approx(60)    # burst rule blocks
    clock.now += 61
    for _ in range(3):
        limiter.record("a"); clock.now += 61                # reach 5 in the hour while respecting the burst rule
    assert limiter.retry_after("a") > 60                    # now the hourly rule is the one that blocks


def test_non_positive_rules_disable_the_limiter():
    limiter, _ = make([(0, 60), (5, 0)])
    assert not limiter.enabled
    for _ in range(100):
        limiter.record("a")
    assert limiter.retry_after("a") == 0


def test_memory_is_bounded():
    limiter, clock = make([(5, 60)], max_keys=50)
    for i in range(500):
        limiter.record(f"ip-{i}")
        clock.now += 0.01
    assert len(limiter._hits) <= 50
    assert "ip-499" in limiter._hits                        # the most recent clients are the ones kept


def test_quiet_clients_are_forgotten():
    limiter, clock = make([(5, 60)], max_keys=2)
    limiter.record("old")
    clock.now += 120
    limiter.record("x"); limiter.record("y")                # exceeding max_keys triggers a clean-up
    assert "old" not in limiter._hits


def test_reset_clears_everything():
    limiter, _ = make([(1, 60)])
    limiter.record("a")
    limiter.reset()
    assert limiter.retry_after("a") == 0


# ---- which address counts as the client ------------------------------------------------------

def test_without_a_configured_header_the_socket_address_is_used_even_if_headers_lie():
    headers = {"x-forwarded-for": "6.6.6.6", "cf-connecting-ip": "7.7.7.7"}
    assert client_ip(headers, "10.0.0.5") == ("10.0.0.5", "peer")


def test_x_forwarded_for_takes_the_entry_added_by_the_trusted_proxy_not_the_forgeable_leftmost():
    # A client sent "1.1.1.1"; the one trusted proxy appended the address it actually saw.
    assert client_ip({"x-forwarded-for": "1.1.1.1, 203.0.113.9"}, "10.0.0.5", "X-Forwarded-For", hops=1) == ("203.0.113.9", "x-forwarded-for")


def test_x_forwarded_for_with_two_trusted_proxies():
    headers = {"x-forwarded-for": "9.9.9.9, 203.0.113.9, 172.16.0.2"}   # client-forged, real client, first proxy
    assert client_ip(headers, "10.0.0.5", "x-forwarded-for", hops=2)[0] == "203.0.113.9"


def test_x_forwarded_for_shorter_than_the_trusted_hops_falls_back_to_the_peer():
    assert client_ip({"x-forwarded-for": "203.0.113.9"}, "10.0.0.5", "x-forwarded-for", hops=2) == ("10.0.0.5", "peer")


def test_single_value_header_such_as_cf_connecting_ip():
    assert client_ip({"cf-connecting-ip": "203.0.113.9"}, "10.0.0.5", "CF-Connecting-IP") == ("203.0.113.9", "cf-connecting-ip")


@pytest.mark.parametrize("bad", ["not-an-ip", "999.1.1.1", "'; DROP TABLE", "  "])
def test_garbage_or_missing_header_values_fall_back_to_the_peer(bad):
    assert client_ip({"cf-connecting-ip": bad}, "10.0.0.5", "cf-connecting-ip") == ("10.0.0.5", "peer")
    assert client_ip({}, "10.0.0.5", "cf-connecting-ip") == ("10.0.0.5", "peer")


def test_ipv6_clients_share_one_allowance_per_slash_64():
    a = normalize_ip("2001:db8:1:2:aaaa:bbbb:cccc:dddd")
    b = normalize_ip("2001:db8:1:2:1111:2222:3333:4444")
    c = normalize_ip("2001:db8:1:3::1")
    assert a == b and a != c and a.endswith("/64")


def test_ipv4_mapped_ipv6_counts_as_the_ipv4_address():
    assert normalize_ip("::ffff:203.0.113.9") == "203.0.113.9"


def test_unknown_peer_does_not_crash():
    assert client_ip({}, None) == ("unknown", "peer")
    assert client_ip({}, "testclient") == ("testclient", "peer")   # Starlette's TestClient reports a name, not an IP
