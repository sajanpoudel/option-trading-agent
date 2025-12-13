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


def test_detector_maps_strong_uptrend_to_strong_trend():
    assert ScenarioDetector().detect({"scenario": "strong_uptrend"}) == "strong_trend"


def test_detector_maps_strong_downtrend_to_strong_trend():
    assert ScenarioDetector().detect({"scenario": "strong_downtrend"}) == "strong_trend"


def test_detector_maps_range_bound_to_range_bound():
    assert ScenarioDetector().detect({"scenario": "range_bound"}) == "range_bound"


def test_detector_maps_breakout_to_high_volatility():
    assert ScenarioDetector().detect({"scenario": "breakout"}) == "high_volatility"


def test_detector_maps_potential_reversal_to_normal():
    assert ScenarioDetector().detect({"scenario": "potential_reversal"}) == "normal"


def test_detector_maps_unheard_of_to_normal():
    assert ScenarioDetector().detect({"scenario": "unheard_of"}) == "normal"


def test_detector_defaults_to_normal_without_data():
    assert ScenarioDetector().detect({}) == "normal"


def test_detector_survives_bad_input():
    assert ScenarioDetector().detect(None) == "normal"


def test_signal_direction_for_a_score_of_0_8(engine):
    signal = engine._generate_signal(0.8, SimpleNamespace(final_score=0.8), {})
    assert signal["direction"] == "STRONG_BUY"


def test_signal_direction_for_a_score_of_0_45(engine):
    signal = engine._generate_signal(0.45, SimpleNamespace(final_score=0.45), {})
    assert signal["direction"] == "BUY"


def test_signal_direction_for_a_score_of_0_0(engine):
    signal = engine._generate_signal(0.0, SimpleNamespace(final_score=0.0), {})
    assert signal["direction"] == "HOLD"


def test_signal_direction_for_a_score_of_minus_0_45(engine):
    signal = engine._generate_signal(-0.45, SimpleNamespace(final_score=-0.45), {})
    assert signal["direction"] == "SELL"


def test_signal_direction_for_a_score_of_minus_0_8(engine):
    signal = engine._generate_signal(-0.8, SimpleNamespace(final_score=-0.8), {})
    assert signal["direction"] == "STRONG_SELL"


def test_signal_averages_the_decision_and_the_ensemble(engine):
    signal = engine._generate_signal(0.8, SimpleNamespace(final_score=0.2), {})
    assert signal["score"] == pytest.approx(0.5)
    assert signal["direction"] == "BUY"


def test_signal_thresholds_are_inclusive(engine):
    assert engine._generate_signal(0.3, SimpleNamespace(final_score=0.3), {})["direction"] == "BUY"
    assert engine._generate_signal(0.6, SimpleNamespace(final_score=0.6), {})["direction"] == "STRONG_BUY"
    assert engine._generate_signal(-0.3, SimpleNamespace(final_score=-0.3), {})["direction"] == "SELL"


@pytest.mark.parametrize(
    "direction, scenario, expected",
    [
        ("BUY", "normal", "long_call"),
        ("STRONG_BUY", "high_volatility", "long_call_spread"),
        ("BUY", "breakout", "long_call_spread"),
        ("SELL", "normal", "long_put"),
        ("STRONG_SELL", "breakout", "long_put_spread"),
        ("HOLD", "range_bound", "iron_condor"),
        ("HOLD", "normal", "straddle"),
    ],
)
def test_options_strategy_choice(engine, direction, scenario, expected):
    assert engine._select_options_strategy(direction, scenario) == expected


def test_reasoning_mentions_the_score_and_scenario(engine):
    text = engine._generate_reasoning({"technical": {"scenario": "breakout"}}, 0.42)
    assert "0.420" in text
    assert "breakout" in text
    assert "upward momentum" in text


def test_reasoning_describes_bearish_and_mixed_scores(engine):
    assert "downward pressure" in engine._generate_reasoning({}, -0.5)
    assert "Mixed signals" in engine._generate_reasoning({}, 0.0)


def test_the_signal_carries_a_strategy_and_reasoning(engine):
    signal = engine._generate_signal(0.5, SimpleNamespace(final_score=0.5), {"technical": {"scenario": "normal"}})
    assert signal["strategy_type"] == "moderate_bullish"
    assert signal["options_strategy"] == "long_call"
    assert signal["reasoning"].startswith("Decision score")


def bullish_signal(score=0.7):
    return {"direction": "BUY", "score": score}


def test_a_buy_signal_gives_calls():
    recs = RiskBasedStrikeSelector().select_strikes(bullish_signal(), "AAPL", {"risk_level": "moderate"})
    assert recs
    assert {r.option_type for r in recs} == {"call"}


def test_a_sell_signal_gives_puts():
    signal = {"direction": "SELL", "score": -0.7}
    recs = RiskBasedStrikeSelector().select_strikes(signal, "AAPL", {"risk_level": "moderate"})
    assert {r.option_type for r in recs} == {"put"}


def test_a_hold_signal_gives_an_iron_condor():
    signal = {"direction": "HOLD", "score": 0.0}
    recs = RiskBasedStrikeSelector().select_strikes(signal, "AAPL", {"risk_level": "moderate"})
    assert [r.option_type for r in recs] == ["iron_condor"]


def test_conservative_profile_caps_the_loss_at_2_percent():
    recs = RiskBasedStrikeSelector().select_strikes(bullish_signal(), "AAPL", {"risk_level": "conservative"})
    assert all(r.max_loss == 0.02 for r in recs)


def test_moderate_profile_caps_the_loss_at_5_percent():
    recs = RiskBasedStrikeSelector().select_strikes(bullish_signal(), "AAPL", {"risk_level": "moderate"})
    assert all(r.max_loss == 0.05 for r in recs)
