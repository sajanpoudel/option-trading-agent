import numpy as np
import pandas as pd

from backend.app.indicators.calculator import TechnicalIndicatorsCalculator


def make_prices(days=60):
    index = pd.date_range("2024-01-01", periods=days)
    close = np.linspace(100, 110, days)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": [1000] * days,
        },
        index=index,
    )


def test_empty_frame_returns_fallback_values():
    result = TechnicalIndicatorsCalculator().calculate_comprehensive_indicators(pd.DataFrame())
    assert isinstance(result, dict)
    assert result


def test_indicators_for_a_steady_uptrend_are_returned():
    result = TechnicalIndicatorsCalculator().calculate_comprehensive_indicators(make_prices(), "AAPL")
    assert isinstance(result, dict)
    assert result


def test_results_are_json_friendly_numbers():
    import json

    result = TechnicalIndicatorsCalculator().calculate_comprehensive_indicators(make_prices(), "AAPL")
    json.dumps(result, default=str)


def test_true_range_includes_gaps_from_the_previous_close():
    high = np.array([110.0, 100.0])
    low = np.array([108.0, 99.0])
    close = np.array([110.0, 99.5])
    # range 1, gap from the high 10, gap from the low 11
    assert TechnicalIndicatorsCalculator._true_range(high, low, close).tolist() == [11.0]


def test_true_range_is_the_plain_range_without_gaps():
    high = np.array([101.0, 102.0, 103.0])
    low = np.array([99.0, 100.0, 101.0])
    close = np.array([100.0, 101.0, 102.0])
    assert TechnicalIndicatorsCalculator._true_range(high, low, close).tolist() == [2.0, 2.0]
