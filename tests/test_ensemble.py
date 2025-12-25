import pytest

from backend.app.ml.ensemble import EnsembleDecisionModel


@pytest.fixture(scope="module")
def model():
    return EnsembleDecisionModel()


RISK_LOW = {"risk_level": "low"}
RISK_HIGH = {"risk_level": "high"}


def test_base_weights_sum_to_one(model):
    assert sum(model.base_weights.values()) == pytest.approx(1.0)
