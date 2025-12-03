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
