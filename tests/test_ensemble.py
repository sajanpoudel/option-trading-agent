import pytest

from backend.app.ml.ensemble import EnsembleDecisionModel


@pytest.fixture(scope="module")
def model():
    return EnsembleDecisionModel()


RISK_LOW = {"risk_level": "low"}
RISK_HIGH = {"risk_level": "high"}


def test_base_weights_sum_to_one(model):
    assert sum(model.base_weights.values()) == pytest.approx(1.0)


def test_high_vol_regime_weights_sum_to_one(model):
    assert sum(model._adjust_weights_for_regime("high_vol").values()) == pytest.approx(1.0)


def test_low_vol_regime_weights_sum_to_one(model):
    assert sum(model._adjust_weights_for_regime("low_vol").values()) == pytest.approx(1.0)


def test_earnings_week_regime_weights_sum_to_one(model):
    assert sum(model._adjust_weights_for_regime("earnings_week").values()) == pytest.approx(1.0)


def test_fomc_week_regime_weights_sum_to_one(model):
    assert sum(model._adjust_weights_for_regime("fomc_week").values()) == pytest.approx(1.0)


def test_normal_regime_weights_sum_to_one(model):
    assert sum(model._adjust_weights_for_regime("normal").values()) == pytest.approx(1.0)


def test_unknown_regime_weights_sum_to_one(model):
    assert sum(model._adjust_weights_for_regime("unknown").values()) == pytest.approx(1.0)


def test_normal_regime_keeps_the_base_weights(model):
    assert model._adjust_weights_for_regime("normal") == pytest.approx(model.base_weights)


def test_adjusting_weights_does_not_change_the_base_weights(model):
    before = dict(model.base_weights)
    model._adjust_weights_for_regime("fomc_week")
    assert model.base_weights == before


def test_fomc_week_shifts_weight_from_technical_to_volatility(model):
    weights = model._adjust_weights_for_regime("fomc_week")
    assert weights["volatility"] > model.base_weights["volatility"]
    assert weights["technical"] < model.base_weights["technical"]


def test_safe_score_clips_to_the_valid_range(model):
    assert model._safe_score(5, 0.0) == 1.0
    assert model._safe_score(-5, 0.0) == -1.0
    assert model._safe_score(0.25, 0.0) == 0.25


def test_safe_score_uses_the_default_for_exceptions_and_other_types(model):
    assert model._safe_score(RuntimeError("model failed"), 0.1) == 0.1
    assert model._safe_score("0.5", 0.2) == 0.2
    assert model._safe_score(None, -0.3) == -0.3


def test_ensemble_score_is_the_weighted_average(model):
    scores = {"sentiment": 1.0, "flow": 0.0}
    weights = {"sentiment": 0.2, "flow": 0.3}
    assert model._calculate_ensemble_score(scores, weights) == pytest.approx(0.4)


def test_ensemble_score_ignores_components_without_a_weight(model):
    scores = {"sentiment": 1.0, "mystery": -1.0}
    assert model._calculate_ensemble_score(scores, {"sentiment": 0.5}) == pytest.approx(1.0)


def test_ensemble_score_with_no_weights_is_zero(model):
    assert model._calculate_ensemble_score({"sentiment": 1.0}, {}) == 0.0


def test_confidence_is_higher_when_models_agree(model):
    weights = model.base_weights
    agree = model._calculate_ensemble_confidence({"a": 0.6, "b": 0.6, "c": 0.6}, weights, "normal")
    split = model._calculate_ensemble_confidence({"a": 0.6, "b": -0.6, "c": 0.0}, weights, "normal")
    assert agree > split


def test_fomc_week_lowers_the_confidence(model):
    scores = {"a": 0.5, "b": 0.5}
    normal = model._calculate_ensemble_confidence(scores, model.base_weights, "normal")
    fomc = model._calculate_ensemble_confidence(scores, model.base_weights, "fomc_week")
    assert fomc < normal


def test_confidence_stays_between_zero_and_one(model):
    for scores in ({"a": 1.0, "b": 1.0}, {"a": 0.0, "b": 0.0}, {"a": -1.0, "b": 1.0}):
        value = model._calculate_ensemble_confidence(scores, model.base_weights, "low_vol")
        assert 0.0 <= value <= 1.0


def test_a_single_model_gets_a_middling_agreement(model):
    value = model._calculate_ensemble_confidence({"a": 0.0}, model.base_weights, "normal")
    assert value == pytest.approx(0.3)


def test_a_score_of_0_7_is_strong_buy(model):
    direction, _ = model._determine_direction_and_strength(0.7, 0.5)
    assert direction == "STRONG_BUY"


def test_a_score_of_0_4_is_buy(model):
    direction, _ = model._determine_direction_and_strength(0.4, 0.5)
    assert direction == "BUY"


def test_a_score_of_0_0_is_hold(model):
    direction, _ = model._determine_direction_and_strength(0.0, 0.5)
    assert direction == "HOLD"


def test_a_score_of_minus_0_4_is_sell(model):
    direction, _ = model._determine_direction_and_strength(-0.4, 0.5)
    assert direction == "SELL"


def test_a_score_of_minus_0_7_is_strong_sell(model):
    direction, _ = model._determine_direction_and_strength(-0.7, 0.5)
    assert direction == "STRONG_SELL"


def test_direction_thresholds_are_inclusive(model):
    assert model._determine_direction_and_strength(0.65, 0.5)[0] == "STRONG_BUY"
    assert model._determine_direction_and_strength(0.30, 0.5)[0] == "BUY"
    assert model._determine_direction_and_strength(-0.30, 0.5)[0] == "SELL"
    assert model._determine_direction_and_strength(-0.65, 0.5)[0] == "STRONG_SELL"


@pytest.mark.parametrize(
    "score, confidence, strength",
    [(0.9, 0.9, "strong"), (0.5, 0.5, "moderate"), (0.1, 0.1, "weak")],
)
def test_strength_follows_score_and_confidence(model, score, confidence, strength):
    assert model._determine_direction_and_strength(score, confidence)[1] == strength


def test_strength_uses_the_size_of_a_negative_score(model):
    assert model._determine_direction_and_strength(-0.9, 0.9)[1] == "strong"


def test_risk_is_low_for_agreeing_models_in_a_normal_market(model):
    scores = {"a": 0.5, "b": 0.5, "c": 0.4, "d": 0.5}
    result = model._assess_ensemble_risk(scores, {}, "normal")
    assert result["risk_level"] == "low"
    assert result["risk_factors"] == []


def test_disagreement_adds_a_risk_factor(model):
    scores = {"a": 1.0, "b": -1.0, "c": 1.0, "d": -1.0}
    result = model._assess_ensemble_risk(scores, {}, "normal")
    assert "High disagreement between models" in result["risk_factors"]


def test_risky_regimes_add_a_risk_factor(model):
    scores = {"a": 0.5, "b": 0.5}
    result = model._assess_ensemble_risk(scores, {}, "fomc_week")
    assert any("fomc_week" in factor for factor in result["risk_factors"])


def test_missing_data_adds_a_data_quality_risk(model):
    scores = {"a": 0.0, "b": 0.0, "c": 0.0, "d": 0.4}
    result = model._assess_ensemble_risk(scores, {}, "normal")
    assert "Low data quality or model failures" in result["risk_factors"]


def test_stacked_risks_become_high_risk_and_reduce_sizing(model):
    scores = {"a": 1.0, "b": -1.0, "c": 0.0, "d": 0.0}
    result = model._assess_ensemble_risk(scores, {"regime": "high_vol"}, "high_vol")
    assert result["risk_level"] == "high"
    assert result["recommendation"] == "Reduce position size"
    assert result["risk_score"] <= 1.0
