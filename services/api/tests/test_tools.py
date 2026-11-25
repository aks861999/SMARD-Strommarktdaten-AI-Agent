from datetime import date

from prices_service.sources import DemoPriceSource  # demo data lives in the prices service (see pytest.ini)
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


def test_dst_days_report_the_real_number_of_hours():
    tools = make_tools()
    spring = tools.get_hourly_prices("2026-03-29")   # clocks jump 02:00 -> 03:00
    autumn = tools.get_hourly_prices("2026-10-25")   # 02:00 occurs twice
    assert (spring["expected_hours"], spring["available_hours"], spring["status"]) == (23, 23, "complete")
    assert (autumn["expected_hours"], autumn["available_hours"], autumn["status"]) == (25, 25, "complete")


def test_estimate_run_cost_spans_spring_forward_correctly():
    # 01:00 + 3 absolute hours covers 01:00, 03:00 and 04:00 local time.
    result = make_tools().estimate_run_cost(
        date="2026-03-29", start_hour=1, duration_hours=3, power_kw=1,
    )
    assert result["status"] == "ok"
    assert len(result["hourly_prices_eur_mwh"]) == 3


def test_estimate_run_cost_uses_both_repeated_autumn_hours():
    # 01:00 + 3 absolute hours covers 01:00 CEST, 02:00 CEST and 02:00 CET.
    result = make_tools().estimate_run_cost(
        date="2026-10-25", start_hour=1, duration_hours=3, power_kw=1,
    )
    assert result["status"] == "ok"
    assert len(result["hourly_prices_eur_mwh"]) == 3


def test_cheapest_window_is_found_on_dst_day():
    result = make_tools().find_cheapest_window(
        date="2026-10-25", duration_hours=2, earliest_hour=0, latest_hour=24,
    )
    assert result["status"] == "ok"
    assert result["start_hour"] in range(0, 24)
