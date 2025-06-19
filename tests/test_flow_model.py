import pytest

from backend.app.ml.flow_model import LightGBMFlowPredictor


@pytest.fixture(scope="module")
def predictor():
    return LightGBMFlowPredictor()
