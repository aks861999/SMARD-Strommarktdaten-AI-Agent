"""SMARD source details owned by the prices service."""
from prices_service.sources import SmardPriceSource


def test_quarterhour_aggregation_requires_four_values_per_hour():
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
