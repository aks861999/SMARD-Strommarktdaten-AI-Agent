"""The prices service, with its route functions called directly (no HTTP client needed)."""
from datetime import date

import pytest
from fastapi import HTTPException, Response

from prices_service.cache import CachedPriceSource
from prices_service.sources import DemoPriceSource
from prices_service.app import create_app, price_payload

DAYS = [date(2026, 11, 16), date(2026, 3, 29), date(2026, 10, 25)]  # normal, 23-hour and 25-hour days


def endpoint(app, path):
    return next(route.endpoint for route in app.routes if getattr(route, "path", None) == path)


# ---- the service ------------------------------------------------------------------------------

def make_app(source=None):
    cache = CachedPriceSource(source or DemoPriceSource())
    return create_app(cache, label="synthetic_demo_data"), cache


def test_prices_endpoint_returns_records_with_a_cache_header():
    app, _ = make_app()
    get = endpoint(app, "/prices")
    first, second = Response(), Response()
    body = get(response=first, target_date=DAYS[0], region="DE-LU")
    assert body["date"] == "2026-11-16" and body["source"] == "synthetic_demo_data" and body["unit"] == "EUR/MWh"
    assert len(body["records"]) == 24 and set(body["records"][0]) == {"timestamp", "price_eur_mwh"}
    assert first.headers["X-Cache"] == "MISS"
    get(response=second, target_date=DAYS[0], region="DE-LU")
    assert second.headers["X-Cache"] == "HIT"


def test_unsupported_region_is_rejected_with_422():
    app, _ = make_app()
    with pytest.raises(HTTPException) as caught:
        endpoint(app, "/prices")(response=Response(), target_date=DAYS[0], region="FR")
    assert caught.value.status_code == 422


def test_an_upstream_failure_becomes_a_502_and_is_not_cached():
    class Broken:
        def get_prices(self, target_date, region="DE-LU"):
            raise RuntimeError("smard timed out")

    app, cache = make_app(Broken())
    with pytest.raises(HTTPException) as caught:
        endpoint(app, "/prices")(response=Response(), target_date=DAYS[0], region="DE-LU")
    assert caught.value.status_code == 502 and "smard timed out" in caught.value.detail
    assert len(cache) == 0


def test_health_reports_source_and_cache_size():
    app, _ = make_app()
    endpoint(app, "/prices")(response=Response(), target_date=DAYS[0], region="DE-LU")
    health = endpoint(app, "/health")()
    assert health["status"] == "ok" and health["source"] == "synthetic_demo_data" and health["cached_days"] == 1
