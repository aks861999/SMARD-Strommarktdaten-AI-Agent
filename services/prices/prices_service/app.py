"""The prices service. One job: serve the hourly day-ahead prices for a date, cached.

    GET /prices?date=YYYY-MM-DD&region=DE-LU
        -> {"date", "region", "source", "unit", "records": [{"timestamp", "price_eur_mwh"}, ...]}
        and an ``X-Cache: HIT|MISS`` header
    GET /health

``PRICE_SOURCE`` is ``smard`` (live, default) or ``demo`` (synthetic CSV). The service holds no secrets and
is meant to be reachable only from the agent API (see docker-compose.yml), never from the internet.
"""
from __future__ import annotations

import logging
import os
from datetime import date

from fastapi import FastAPI, HTTPException, Query, Response

from .cache import CachedPriceSource
from .sources import DemoPriceSource, HourlyPrice, SmardPriceSource

logger = logging.getLogger("uvicorn.error")
SUPPORTED_REGION = "DE-LU"


def source_kind() -> str:
    kind = os.getenv("PRICE_SOURCE", "smard").strip().lower()
    if kind not in {"smard", "demo"}:
        raise ValueError(f"PRICE_SOURCE must be 'smard' or 'demo', got '{kind}'.")
    return kind


def price_payload(prices: list[HourlyPrice], target_date: date, region: str, label: str) -> dict:
    return {
        "date": target_date.isoformat(),
        "region": region,
        "source": label,
        "unit": "EUR/MWh",
        "records": [{"timestamp": p.timestamp.isoformat(), "price_eur_mwh": p.price_eur_mwh}
                    for p in sorted(prices, key=lambda p: p.timestamp)],
    }


def create_app(cache: CachedPriceSource | None = None, label: str | None = None) -> FastAPI:
    kind = source_kind() if cache is None else "custom"
    if cache is None:  # not `cache or ...`: an empty cache has len() 0 and would count as "missing"
        cache = CachedPriceSource(DemoPriceSource() if kind == "demo" else SmardPriceSource())
    label = label or ("synthetic_demo_data" if kind == "demo" else "SMARD")
    app = FastAPI(title="Energy prices service", version="1.0.0")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "service": "energy-prices", "source": label, "cached_days": len(cache)}

    @app.get("/prices")
    def prices(response: Response, target_date: date = Query(..., alias="date"),
               region: str = Query(SUPPORTED_REGION)) -> dict:
        if region != SUPPORTED_REGION:
            raise HTTPException(status_code=422, detail=f"Only the {SUPPORTED_REGION} bidding zone is supported.")
        try:
            records, status = cache.lookup(target_date, region)
        except Exception as exc:  # any upstream failure becomes a clean 502 for the caller
            logger.warning("price source failed for %s: %s", target_date, exc)
            raise HTTPException(status_code=502, detail=f"The price source is unavailable: {exc}") from exc
        response.headers["X-Cache"] = status
        return price_payload(records, target_date, region, label)

    return app


app = create_app()
