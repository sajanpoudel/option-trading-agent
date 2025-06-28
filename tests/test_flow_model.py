import pytest

from backend.app.ml.flow_model import LightGBMFlowPredictor


@pytest.fixture(scope="module")
def predictor():
    return LightGBMFlowPredictor()


def test_unusual_score_is_zero_for_quiet_flow(predictor):
    assert predictor._calculate_unusual_activity_score({}) == 0.0


def test_unusual_score_rises_with_volume(predictor):
    quiet = predictor._calculate_unusual_activity_score({"volume_ratio": 1.0})
    busy = predictor._calculate_unusual_activity_score({"volume_ratio": 4.0})
    assert busy > quiet
