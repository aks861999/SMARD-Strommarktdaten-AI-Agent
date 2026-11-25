"""How the agent API gets prices: the data contract, plus two ways to obtain them.

* ``HttpPriceSource`` asks the separate prices service (docker compose sets ``PRICES_URL``).
* ``local_price_source`` runs the SMARD / demo code inside this process. That code belongs to the prices
  service (``services/prices``), so it exists only in a full checkout of the repository: which is how the
  single-service deployment (Render) and local development run. A container built from ``services/api``
  alone does not contain it and does not need it.

The tools only rely on the ``PriceSource`` interface and on the attributes of ``HourlyPrice``.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Protocol
from zoneinfo import ZoneInfo

import requests

BERLIN = ZoneInfo("Europe/Berlin")
# services/prices, next to services/api, when running from a checkout of the repository.
PRICES_SERVICE_DIR = Path(__file__).resolve().parents[2] / "prices"


@dataclass(frozen=True)
class HourlyPrice:
    """One hourly day-ahead price observation (the same shape the prices service returns)."""

    timestamp: datetime
    price_eur_mwh: float

    def as_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "hour": self.timestamp.hour,
            "price_eur_mwh": round(self.price_eur_mwh, 4),
        }


class PriceSource(Protocol):
    def get_prices(self, target_date: date, region: str = "DE-LU") -> list[HourlyPrice]:
        """Return available hourly prices for one local calendar day."""


def day_hours(target_date: date) -> int:
    """Length of a Berlin calendar day in hours: 23, 24 or 25 around DST changes."""
    start = datetime.combine(target_date, time.min, tzinfo=BERLIN)
    end = datetime.combine(target_date + timedelta(days=1), time.min, tzinfo=BERLIN)
    return int((end.astimezone(timezone.utc) - start.astimezone(timezone.utc)).total_seconds() // 3600)


class HttpPriceSource:
    """Reads prices from the separate prices service (see docker-compose.yml) instead of fetching them here.

    It implements the same ``PriceSource`` interface as the in-process sources, so the tools do not know
    which one they are using. Failures raise ``RuntimeError`` with a readable message; the agent turns
    that into a tool error the model can see.
    """

    def __init__(self, base_url: str, timeout_seconds: int = 60) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "EnergyPricePlanningAgent/1.0"})
        # What the service says its data is ("SMARD" or "synthetic_demo_data"); used to label tool results.
        self.source_label = "SMARD"

    def get_prices(self, target_date: date, region: str = "DE-LU") -> list[HourlyPrice]:
        try:
            response = self.session.get(f"{self.base_url}/prices", timeout=self.timeout_seconds,
                                        params={"date": target_date.isoformat(), "region": region})
        except requests.RequestException as exc:
            raise RuntimeError(f"The price service is unreachable at {self.base_url}: {exc}") from exc
        if response.status_code != 200:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            raise RuntimeError(f"The price service answered HTTP {response.status_code}: {str(detail)[:300]}")
        payload = response.json()
        self.source_label = payload.get("source") or self.source_label
        return [
            HourlyPrice(timestamp=datetime.fromisoformat(item["timestamp"]).astimezone(BERLIN),
                        price_eur_mwh=float(item["price_eur_mwh"]))
            for item in payload["records"]
        ]


def local_price_source(kind: str) -> PriceSource:
    """A price source that runs in this process: ``smard`` (live) or ``demo`` (synthetic CSV)."""
    if not PRICES_SERVICE_DIR.is_dir():
        raise RuntimeError(
            "No price source is configured: set PRICES_URL to the prices service, or run from a full "
            "checkout of the repository (services/prices is missing)."
        )
    if str(PRICES_SERVICE_DIR) not in sys.path:
        sys.path.insert(0, str(PRICES_SERVICE_DIR))
    from prices_service.sources import DemoPriceSource, SmardPriceSource  # owned by the prices service

    return DemoPriceSource() if kind == "demo" else SmardPriceSource()
