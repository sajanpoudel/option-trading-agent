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
