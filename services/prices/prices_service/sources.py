"""Price-data sources: a deterministic local demo source and the SMARD API.

Owned by the prices service. The agent API never imports this in a container; it asks the service over HTTP."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Protocol
from zoneinfo import ZoneInfo

import requests

BERLIN = ZoneInfo("Europe/Berlin")
SMARD_BASE_URL = "https://www.smard.de/app/chart_data"
DAY_AHEAD_FILTER = "4169"


@dataclass(frozen=True)
class HourlyPrice:
    """One hourly day-ahead price observation."""

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


class DemoPriceSource:
    """Loads clearly labelled synthetic values so demos never depend on the network."""

    def __init__(self, csv_path: Path | None = None) -> None:
        self.csv_path = csv_path or Path(__file__).resolve().parent.parent / "data" / "sample_prices.csv"
        with self.csv_path.open("r", newline="", encoding="utf-8") as file:
            self._rows = list(csv.DictReader(file))

    def get_prices(self, target_date: date, region: str = "DE-LU") -> list[HourlyPrice]:
        if region != "DE-LU":
            raise ValueError("The demo dataset only supports the DE-LU bidding zone.")

        # Walk the local day in absolute hours so DST days return 23 or 25 rows
        # instead of a hard-coded 24. The synthetic price is keyed by local hour.
        by_hour = {int(row["hour"]): float(row["price_eur_mwh"]) for row in self._rows}
        local_start = datetime.combine(target_date, time.min, tzinfo=BERLIN)
        start_utc = local_start.astimezone(timezone.utc)
        end_utc = (local_start + timedelta(days=1)).astimezone(timezone.utc)

        results: list[HourlyPrice] = []
        cursor = start_utc
        while cursor < end_utc:
            local_timestamp = cursor.astimezone(BERLIN)
            price = by_hour.get(local_timestamp.hour)
            if price is not None:
                results.append(HourlyPrice(timestamp=local_timestamp, price_eur_mwh=price))
            cursor += timedelta(hours=1)
        return results


class SmardPriceSource:
    """Fetches hourly day-ahead prices from the public SMARD chart-data API.

    SMARD publishes weekly time-series chunks. This client reads the index,
    downloads only chunks overlapping the requested Berlin calendar day, and
    filters observations to that local date.
    """

    def __init__(self, timeout_seconds: int = 20) -> None:
        self.timeout_seconds = timeout_seconds
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "EnergyPricePlanningAgent/1.0"})
        self._index_cache: dict[tuple[str, str], list[int]] = {}
        self._chunk_cache: dict[tuple[str, str, int], list[tuple[int, float]]] = {}

    def _get_json(self, url: str) -> dict:
        response = self.session.get(url, timeout=self.timeout_seconds)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError(f"Unexpected JSON response from SMARD: {url}")
        return payload

    def _get_index(self, region: str, resolution: str) -> list[int]:
        cache_key = (region, resolution)
        if cache_key not in self._index_cache:
            url = f"{SMARD_BASE_URL}/{DAY_AHEAD_FILTER}/{region}/index_{resolution}.json"
            payload = self._get_json(url)
            timestamps = payload.get("timestamps")
            if not isinstance(timestamps, list):
                raise ValueError("SMARD index response did not contain a 'timestamps' list.")
            self._index_cache[cache_key] = [int(value) for value in timestamps]
        return self._index_cache[cache_key]

    def _get_week_chunk(self, region: str, resolution: str, chunk_timestamp: int) -> list[tuple[int, float]]:
        cache_key = (region, resolution, chunk_timestamp)
        if cache_key not in self._chunk_cache:
            url = (
                f"{SMARD_BASE_URL}/{DAY_AHEAD_FILTER}/{region}/"
                f"{DAY_AHEAD_FILTER}_{region}_{resolution}_{chunk_timestamp}.json"
            )
            payload = self._get_json(url)
            series = payload.get("series")
            if not isinstance(series, list):
                raise ValueError("SMARD time-series response did not contain a 'series' list.")

            points: list[tuple[int, float]] = []
            for item in series:
                if not isinstance(item, list) or len(item) < 2 or item[1] is None:
                    continue
                points.append((int(item[0]), float(item[1])))
            self._chunk_cache[cache_key] = points
        return self._chunk_cache[cache_key]

    @staticmethod
    def _aggregate_quarterhours(observations: dict[int, float]) -> dict[int, float]:
        """Convert complete sets of four 15-minute prices into hourly averages.

        The grouping key is a UTC hour boundary. This avoids ambiguous local-hour
        keys on the daylight-saving clock-change day in Europe/Berlin.
        """
        groups: dict[int, list[float]] = {}
        for timestamp_ms, price in observations.items():
            utc_hour_ms = (timestamp_ms // 3_600_000) * 3_600_000
            groups.setdefault(utc_hour_ms, []).append(price)
        return {
            hour_timestamp: sum(values) / 4
            for hour_timestamp, values in groups.items()
            if len(values) == 4
        }

    def get_prices(self, target_date: date, region: str = "DE-LU") -> list[HourlyPrice]:
        # SMARD switched the day-ahead market from hourly to quarter-hourly
        # products on 2025-10-01. Aggregate four quarter-hours for this hourly agent.
        resolution = "quarterhour" if target_date >= date(2025, 10, 1) else "hour"
        index_timestamps = self._get_index(region, resolution)
        if not index_timestamps:
            return []

        local_start = datetime.combine(target_date, time.min, tzinfo=BERLIN)
        local_end = local_start + timedelta(days=1)
        target_start_ms = int(local_start.astimezone(timezone.utc).timestamp() * 1000)
        target_end_ms = int(local_end.astimezone(timezone.utc).timestamp() * 1000)

        # A SMARD chunk is a weekly block. Select every indexed block that overlaps
        # the requested local day, rather than assuming the week starts at midnight.
        overlapping_chunks = []
        for chunk_timestamp in index_timestamps:
            chunk_end = chunk_timestamp + 7 * 24 * 60 * 60 * 1000
            if chunk_timestamp < target_end_ms and chunk_end > target_start_ms:
                overlapping_chunks.append(chunk_timestamp)

        observations: dict[int, float] = {}
        for chunk_timestamp in overlapping_chunks:
            for timestamp_ms, price in self._get_week_chunk(region, resolution, chunk_timestamp):
                if target_start_ms <= timestamp_ms < target_end_ms:
                    observations[timestamp_ms] = price

        if resolution == "quarterhour":
            observations = self._aggregate_quarterhours(observations)

        results = [
            HourlyPrice(
                timestamp=datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).astimezone(BERLIN),
                price_eur_mwh=price,
            )
            for timestamp_ms, price in sorted(observations.items())
        ]
        return results
