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
