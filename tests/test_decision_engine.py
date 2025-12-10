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
    assert weights.flow > engine.base_weights.flow


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


def test_numeric_sentiment_passes_through(engine):
    assert engine._sentiment_to_numeric(0.25) == 0.25
    assert engine._sentiment_to_numeric(1) == 1.0


def test_unknown_sentiment_is_neutral(engine):
    assert engine._sentiment_to_numeric("confused") == 0.0
    assert engine._sentiment_to_numeric(None) == 0.0


def test_scores_are_clamped_to_plus_minus_one(engine):
    scores = engine._normalize_scores({"a": 5, "b": -3, "c": 0.4})
    assert scores == {"a": 1.0, "b": -1.0, "c": 0.4}


def test_empty_agent_results_give_a_neutral_decision(engine):
    assert engine._calculate_weighted_decision({}, engine.base_weights) == 0.0


def test_weighted_decision_uses_each_agent_score(engine):
    results = {
        "technical": {"weighted_score": 1.0},
        "sentiment": {"overall_sentiment": "bullish"},
        "flow": {"flow_sentiment": "bearish"},
        "history": {"pattern_strength": 0.5},
    }
    expected = 1.0 * 0.60 + 0.5 * 0.10 + -0.5 * 0.10 + 0.5 * 0.20
    assert engine._calculate_weighted_decision(results, engine.base_weights) == pytest.approx(expected)


def test_a_broken_score_falls_back_to_zero(engine):
    results = {"technical": {"weighted_score": "not a number"}}
    assert engine._calculate_weighted_decision(results, engine.base_weights) == 0.0
