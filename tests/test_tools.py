from datetime import date

from energy_agent.sources import DemoPriceSource
from energy_agent.tools import EnergyTools


def make_tools() -> EnergyTools:
    return EnergyTools(DemoPriceSource())


def test_demo_source_returns_24_hours():
    result = make_tools().get_hourly_prices("2026-11-16")
    assert result["available_hours"] == 24
    assert result["status"] == "complete"
    assert result["source"] == "synthetic_demo_data"


def test_analyze_prices_returns_statistics():
    result = make_tools().analyze_prices("2026-11-16")
    assert result["observations"] == 24
    assert result["minimum_eur_mwh"] == 34
    assert result["maximum_eur_mwh"] == 126


def test_cheapest_two_hour_window_in_requested_range():
    result = make_tools().find_cheapest_window(
        date="2026-11-16",
        duration_hours=2,
        earliest_hour=8,
        latest_hour=22,
    )
    assert result["status"] == "ok"
    assert result["start_hour"] == 11
    assert result["hourly_prices_eur_mwh"] == [38.0, 34.0]


def test_estimate_wholesale_run_cost():
    result = make_tools().estimate_run_cost(
        date="2026-11-16",
        start_hour=11,
        duration_hours=2,
        power_kw=2,
    )
    # 2 kW × (38 + 34) EUR/MWh ÷ 1,000 = EUR 0.144
    assert result["status"] == "ok"
    assert result["energy_kwh"] == 4
    assert result["estimated_wholesale_energy_cost_eur"] == 0.144


def test_quarterhour_aggregation_requires_four_values_per_hour():
    from energy_agent.sources import SmardPriceSource

    hour_start = 1_800_000_000_000
    quarterhour_values = {
        hour_start: 40.0,
        hour_start + 900_000: 50.0,
        hour_start + 1_800_000: 60.0,
        hour_start + 2_700_000: 70.0,
        hour_start + 3_600_000: 999.0,  # incomplete next hour: must be dropped
    }
    result = SmardPriceSource._aggregate_quarterhours(quarterhour_values)
    assert result == {hour_start: 55.0}
