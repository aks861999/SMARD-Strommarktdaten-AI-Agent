"""The four tools exposed to the LLM."""

from __future__ import annotations

import statistics
from datetime import date, datetime, timedelta
from typing import Any, Callable

from .sources import HourlyPrice, PriceSource, day_hours as _day_hours


def _epoch(timestamp: datetime) -> int:
    """Return absolute epoch seconds so DST shifts never masquerade as extra hours."""
    return int(timestamp.timestamp())


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_hourly_prices",
            "description": (
                "Retrieve the available hourly German day-ahead electricity prices for one date. "
                "Use this for questions about actual prices at particular hours or for price data "
                "that you need to inspect. Prices are in EUR/MWh."
            ),
            "parameters": {
                "type": "object",
                "required": ["date"],
                "properties": {
                    "date": {"type": "string", "description": "Date in YYYY-MM-DD format."},
                    "region": {"type": "string", "enum": ["DE-LU"], "description": "SMARD bidding zone; use DE-LU."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_prices",
            "description": (
                "Calculate descriptive statistics for one day's available hourly electricity prices: "
                "minimum, maximum, mean, median, range, and cheapest/most expensive hours. "
                "Use this for analysis or volatility questions."
            ),
            "parameters": {
                "type": "object",
                "required": ["date"],
                "properties": {
                    "date": {"type": "string", "description": "Date in YYYY-MM-DD format."},
                    "region": {"type": "string", "enum": ["DE-LU"], "description": "SMARD bidding zone; use DE-LU."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_cheapest_window",
            "description": (
                "Find the cheapest consecutive run window on one date. The window must fit between "
                "earliest_hour (inclusive) and latest_hour (exclusive). Use this for appliance scheduling. "
                "Only complete consecutive hourly windows are considered."
            ),
            "parameters": {
                "type": "object",
                "required": ["date", "duration_hours"],
                "properties": {
                    "date": {"type": "string", "description": "Date in YYYY-MM-DD format."},
                    "duration_hours": {"type": "integer", "minimum": 1, "maximum": 12},
                    "earliest_hour": {"type": "integer", "minimum": 0, "maximum": 23, "description": "Earliest permitted start hour, inclusive. Default 0."},
                    "latest_hour": {"type": "integer", "minimum": 1, "maximum": 24, "description": "Latest permitted end hour, exclusive. Default 24."},
                    "region": {"type": "string", "enum": ["DE-LU"], "description": "SMARD bidding zone; use DE-LU."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "estimate_run_cost",
            "description": (
                "Estimate the wholesale energy cost of operating an appliance at a given start hour. "
                "Assumes constant power consumption for each hour. This is only the day-ahead wholesale "
                "energy component, not the complete household electricity bill."
            ),
            "parameters": {
                "type": "object",
                "required": ["date", "start_hour", "duration_hours", "power_kw"],
                "properties": {
                    "date": {"type": "string", "description": "Date in YYYY-MM-DD format."},
                    "start_hour": {"type": "integer", "minimum": 0, "maximum": 23},
                    "duration_hours": {"type": "integer", "minimum": 1, "maximum": 12},
                    "power_kw": {"type": "number", "exclusiveMinimum": 0, "description": "Appliance power in kW."},
                    "region": {"type": "string", "enum": ["DE-LU"], "description": "SMARD bidding zone; use DE-LU."},
                },
            },
        },
    },
]


class EnergyTools:
    """Implements the four registered energy-price tools."""

    def __init__(self, price_source: PriceSource) -> None:
        self.price_source = price_source
        self.functions: dict[str, Callable[..., dict[str, Any]]] = {
            "get_hourly_prices": self.get_hourly_prices,
            "analyze_prices": self.analyze_prices,
            "find_cheapest_window": self.find_cheapest_window,
            "estimate_run_cost": self.estimate_run_cost,
        }

    @staticmethod
    def _parse_date(value: str) -> date:
        try:
            return date.fromisoformat(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("date must be a valid YYYY-MM-DD date") from exc

    def _load(self, date_text: str, region: str = "DE-LU") -> tuple[date, list[HourlyPrice]]:
        target_date = self._parse_date(date_text)
        if region != "DE-LU":
            raise ValueError("Only the DE-LU bidding zone is supported in this project.")
        prices = self.price_source.get_prices(target_date, region)
        return target_date, sorted(prices, key=lambda item: item.timestamp)

    @staticmethod
    def _source_name(source: PriceSource) -> str:
        # A source that fetches remotely (HttpPriceSource) reports what the service told it.
        label = getattr(source, "source_label", None)
        if label:
            return label
        return "synthetic_demo_data" if source.__class__.__name__ == "DemoPriceSource" else "SMARD"

    def get_hourly_prices(self, date: str, region: str = "DE-LU") -> dict[str, Any]:
        target_date, prices = self._load(date, region)
        expected_hours = _day_hours(target_date)
        return {
            "date": target_date.isoformat(),
            "region": region,
            "source": self._source_name(self.price_source),
            "unit": "EUR/MWh",
            "available_hours": len(prices),
            "expected_hours": expected_hours,
            "status": "complete" if len(prices) == expected_hours else "partial_or_unavailable",
            "records": [item.as_dict() for item in prices],
            "note": (
                "Demo values are synthetic and are not real market observations."
                if self._source_name(self.price_source) == "synthetic_demo_data"
                else "Only observations returned by SMARD are included; missing hours are not estimated."
            ),
        }

    def analyze_prices(self, date: str, region: str = "DE-LU") -> dict[str, Any]:
        target_date, prices = self._load(date, region)
        if not prices:
            return {"date": target_date.isoformat(), "status": "no_data", "message": "No prices are available for this date."}

        values = [item.price_eur_mwh for item in prices]
        min_value = min(values)
        max_value = max(values)
        cheapest = [item.timestamp.isoformat() for item in prices if item.price_eur_mwh == min_value]
        most_expensive = [item.timestamp.isoformat() for item in prices if item.price_eur_mwh == max_value]
        mean_value = statistics.mean(values)
        return {
            "date": target_date.isoformat(),
            "region": region,
            "source": self._source_name(self.price_source),
            "observations": len(values),
            "mean_eur_mwh": round(mean_value, 2),
            "median_eur_mwh": round(statistics.median(values), 2),
            "minimum_eur_mwh": round(min_value, 2),
            "minimum_at": cheapest,
            "maximum_eur_mwh": round(max_value, 2),
            "maximum_at": most_expensive,
            "range_eur_mwh": round(max_value - min_value, 2),
            "population_std_dev_eur_mwh": round(statistics.pstdev(values), 2) if len(values) > 1 else 0.0,
            "expected_hours": _day_hours(target_date),
            "status": "complete" if len(values) == _day_hours(target_date) else "partial_data",
        }

    def find_cheapest_window(
        self,
        date: str,
        duration_hours: int,
        earliest_hour: int = 0,
        latest_hour: int = 24,
        region: str = "DE-LU",
    ) -> dict[str, Any]:
        target_date, prices = self._load(date, region)
        if not 1 <= duration_hours <= 12:
            raise ValueError("duration_hours must be between 1 and 12")
        if not 0 <= earliest_hour <= 23:
            raise ValueError("earliest_hour must be between 0 and 23")
        if not 1 <= latest_hour <= 24 or latest_hour <= earliest_hour:
            raise ValueError("latest_hour must be between earliest_hour + 1 and 24")

        candidates: list[tuple[float, list[HourlyPrice]]] = []
        for index in range(0, len(prices) - duration_hours + 1):
            window = prices[index : index + duration_hours]
            # A one-day, consecutive window: every observation must fall on the
            # requested local date and inside the requested local-hour bounds.
            if any(item.timestamp.date() != target_date for item in window):
                continue
            # Keep the search inside the requested daily time boundaries.
            if any(not (earliest_hour <= item.timestamp.hour < latest_hour) for item in window):
                continue
            # Consecutiveness is checked in absolute time, so the two 02:00 hours on
            # the autumn DST day are both valid and the skipped spring hour is not.
            if any(_epoch(window[offset + 1].timestamp) - _epoch(window[offset].timestamp) != 3600
                   for offset in range(duration_hours - 1)):
                continue
            average_price = statistics.mean(item.price_eur_mwh for item in window)
            candidates.append((average_price, window))

        if not candidates:
            return {
                "date": target_date.isoformat(),
                "status": "no_complete_window",
                "message": "No complete consecutive price window was available within those limits.",
            }

        average_price, best_window = min(candidates, key=lambda item: item[0])
        start = best_window[0].timestamp
        end = best_window[-1].timestamp + timedelta(hours=1)
        return {
            "date": target_date.isoformat(),
            "region": region,
            "source": self._source_name(self.price_source),
            "status": "ok",
            "duration_hours": duration_hours,
            "start_at": start.isoformat(),
            "end_at": end.isoformat(),
            "start_hour": start.hour,
            "average_price_eur_mwh": round(average_price, 2),
            "hourly_prices_eur_mwh": [round(item.price_eur_mwh, 2) for item in best_window],
            "hours_evaluated": len(candidates),
            "available_observations": len(prices),
            "full_day_coverage": len(prices) == 24,
            "coverage_note": None if len(prices) == 24 else "The result is the cheapest window among returned observations only; missing hours may contain a cheaper window.",
        }

    def estimate_run_cost(
        self,
        date: str,
        start_hour: int,
        duration_hours: int,
        power_kw: float,
        region: str = "DE-LU",
    ) -> dict[str, Any]:
        target_date, prices = self._load(date, region)
        if not 0 <= start_hour <= 23:
            raise ValueError("start_hour must be between 0 and 23")
        if not 1 <= duration_hours <= 12:
            raise ValueError("duration_hours must be between 1 and 12")
        if power_kw <= 0:
            raise ValueError("power_kw must be greater than zero")

        matching = [item for item in prices if item.timestamp.hour == start_hour]
        if not matching:
            return {"date": target_date.isoformat(), "status": "missing_start_hour", "message": "No price is available for the requested start hour."}
        start = matching[0].timestamp
        # Step through absolute hours rather than adding a wall-clock timedelta:
        # on DST days the local hour labels repeat or skip, the absolute hour does not.
        by_epoch = {_epoch(item.timestamp): item for item in prices}
        start_epoch = _epoch(start)
        selected = [by_epoch.get(start_epoch + 3600 * offset) for offset in range(duration_hours)]
        if any(item is None for item in selected):
            return {"date": target_date.isoformat(), "status": "incomplete_window", "message": "Not enough consecutive hourly prices are available to calculate this cost."}

        hourly = [item for item in selected if item is not None]
        energy_kwh = power_kw * duration_hours
        cost_eur = sum(power_kw * item.price_eur_mwh / 1000 for item in hourly)
        return {
            "date": target_date.isoformat(),
            "region": region,
            "source": self._source_name(self.price_source),
            "status": "ok",
            "start_at": start.isoformat(),
            "duration_hours": duration_hours,
            "power_kw": power_kw,
            "energy_kwh": round(energy_kwh, 3),
            "hourly_prices_eur_mwh": [round(item.price_eur_mwh, 2) for item in hourly],
            "estimated_wholesale_energy_cost_eur": round(cost_eur, 4),
            "explanation": "Energy cost = sum(power in kW × hourly price in EUR/MWh ÷ 1,000).",
            "important_limitation": "Excludes taxes, grid charges, supplier fees and other retail-bill components.",
        }
