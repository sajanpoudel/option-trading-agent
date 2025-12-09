from types import SimpleNamespace

import pytest

from backend.app.core.decision_engine import (
    DecisionEngine,
    RiskBasedStrikeSelector,
    ScenarioDetector,
    ScenarioWeights,
)


@pytest.fixture(scope="module")
def engine():
    return DecisionEngine()


def total(weights: ScenarioWeights) -> float:
    return weights.technical + weights.sentiment + weights.flow + weights.history


def test_base_weights_follow_the_documented_split(engine):
    assert engine.base_weights == ScenarioWeights(0.60, 0.10, 0.10, 0.20)


def test_normal_weights_sum_to_one(engine):
    weights = engine._adjust_weights_for_scenario("normal")
    assert total(weights) == pytest.approx(1.0)


def test_high_volatility_weights_sum_to_one(engine):
    weights = engine._adjust_weights_for_scenario("high_volatility")
    assert total(weights) == pytest.approx(1.0)


def test_low_volatility_weights_sum_to_one(engine):
    weights = engine._adjust_weights_for_scenario("low_volatility")
    assert total(weights) == pytest.approx(1.0)


def test_earnings_approaching_weights_sum_to_one(engine):
    weights = engine._adjust_weights_for_scenario("earnings_approaching")
    assert total(weights) == pytest.approx(1.0)


def test_strong_trend_weights_sum_to_one(engine):
    weights = engine._adjust_weights_for_scenario("strong_trend")
    assert total(weights) == pytest.approx(1.0)


def test_range_bound_weights_sum_to_one(engine):
    weights = engine._adjust_weights_for_scenario("range_bound")
    assert total(weights) == pytest.approx(1.0)


def test_high_volatility_favours_technical_and_flow(engine):
    base = engine.base_weights
    weights = engine._adjust_weights_for_scenario("high_volatility")
    assert weights.technical > base.technical
    assert weights.flow > base.flow
    assert weights.sentiment < base.sentiment


def test_earnings_approaching_favours_options_flow(engine):
    weights = engine._adjust_weights_for_scenario("earnings_approaching")
    assert weights.flow == max(weights.technical, weights.sentiment, weights.flow, weights.history) or weights.flow > engine.base_weights.flow


def test_unknown_scenario_keeps_the_base_weights(engine):
    assert engine._adjust_weights_for_scenario("something else") == engine.base_weights


def test_adjusting_weights_does_not_change_the_base_weights(engine):
    before = ScenarioWeights(**engine.base_weights.__dict__)
    engine._adjust_weights_for_scenario("high_volatility")
    assert engine.base_weights == before


@pytest.mark.parametrize(
    "value, expected",
    [("very_bearish", -1.0), ("bearish", -0.5), ("neutral", 0.0), ("bullish", 0.5), ("very_bullish", 1.0)],
)
def test_sentiment_labels_map_to_numbers(engine, value, expected):
    assert engine._sentiment_to_numeric(value) == expected


def test_sentiment_labels_ignore_case(engine):
    assert engine._sentiment_to_numeric("BULLISH") == 0.5
