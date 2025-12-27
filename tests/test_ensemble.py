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
