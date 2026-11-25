"""Admission control in api.py, tested directly (no HTTP client needed)."""
import pytest
from fastapi import HTTPException

import api

DEFAULTS = {
    "RATE_LIMIT_PER_MINUTE": "0", "RATE_LIMIT_PER_HOUR": "0",
    "RATE_LIMIT_GLOBAL_PER_HOUR": "0", "RATE_LIMIT_MAX_ACTIVE_PER_IP": "0",
}


@pytest.fixture
def configure(monkeypatch):
    """configure(PER_MINUTE=3, ...) sets limits (everything else off) and starts from a clean slate."""
    def apply(**limits):
        for name, value in {**DEFAULTS, **{f"RATE_LIMIT_{k}": str(v) for k, v in limits.items()}}.items():
            monkeypatch.setenv(name, value)
        api._reset_rate_limits()
        api._JOBS.clear()
    yield apply
    monkeypatch.undo()
    api._reset_rate_limits()
    api._JOBS.clear()


def admit(ip="1.1.1.1"):
    api._admit(ip, "peer")


def rejected(ip="1.1.1.1"):
    with pytest.raises(HTTPException) as caught:
        admit(ip)
    assert caught.value.status_code == 429
    return caught.value


def test_per_ip_limit_returns_429_with_a_retry_after_header(configure):
    configure(PER_MINUTE=3)
    for _ in range(3):
        admit()
    error = rejected()
    assert 1 <= int(error.headers["Retry-After"]) <= 60
    assert "wait" in error.detail.lower()


def test_other_clients_are_not_affected_by_one_noisy_client(configure):
    configure(PER_MINUTE=1)
    admit("1.1.1.1")
    rejected("1.1.1.1")
    admit("2.2.2.2")


def test_global_limit_holds_even_when_every_request_claims_a_different_ip(configure):
    configure(GLOBAL_PER_HOUR=2)
    admit("1.1.1.1")
    admit("2.2.2.2")
    assert "busy" in rejected("3.3.3.3").detail.lower()    # a forged, never-seen-before address gets nowhere


def test_a_rejected_request_spends_no_budget_anywhere(configure):
    configure(PER_MINUTE=1, GLOBAL_PER_HOUR=2)
    admit("1.1.1.1")                                        # global: 1 of 2
    rejected("1.1.1.1")                                     # blocked per-IP; must NOT use up global budget
    admit("2.2.2.2")                                        # global: 2 of 2 (fails if the rejection above was counted)
    rejected("3.3.3.3")


def test_active_run_cap_blocks_a_second_run_until_the_first_ends_or_is_cancelled(configure):
    configure(MAX_ACTIVE_PER_IP=1)
    running = api.AgentJob("job-1", "", "1.1.1.1")
    api._JOBS["job-1"] = running
    assert "still running" in rejected("1.1.1.1").detail
    admit("2.2.2.2")                                        # someone else is fine
    running.cancel_event.set()                              # Stop pressed: a cancelled run no longer blocks
    admit("1.1.1.1")
    running.cancel_event.clear()
    running.finished = True                                 # a finished run never blocks
    admit("1.1.1.1")


def test_resuming_an_existing_request_is_neither_counted_nor_blocked(configure):
    configure(PER_MINUTE=1)
    calls = []

    def counting_admit():
        calls.append(1)
        admit()

    job, created = api._get_or_create_job("req-1", "sig", "1.1.1.1", admit=counting_admit)
    assert created and len(calls) == 1
    again, created_again = api._get_or_create_job("req-1", "sig", "1.1.1.1", admit=counting_admit)
    assert again is job and not created_again and len(calls) == 1   # admission was not consulted, no 429


def test_a_rejected_creation_leaves_no_job_behind(configure):
    configure(PER_MINUTE=1)
    api._get_or_create_job("req-1", "sig", "1.1.1.1", admit=admit)
    with pytest.raises(HTTPException):
        api._get_or_create_job("req-2", "sig", "1.1.1.1", admit=admit)
    assert "req-2" not in api._JOBS


def test_limits_default_to_sensible_values_and_can_be_switched_off(monkeypatch):
    for name in DEFAULTS:
        monkeypatch.delenv(name, raising=False)
    api._reset_rate_limits()
    assert api._IP_LIMITER.rules == [(6, 60.0), (60, 3600.0)]
    assert api._GLOBAL_LIMITER.rules == [(300, 3600.0)]
    for name in DEFAULTS:
        monkeypatch.setenv(name, "0")
    api._reset_rate_limits()
    assert not api._IP_LIMITER.enabled and not api._GLOBAL_LIMITER.enabled
    monkeypatch.undo()
    api._reset_rate_limits()
