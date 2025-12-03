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


def test_volume_part_of_the_score_is_capped(predictor):
    assert predictor._calculate_unusual_activity_score({"volume_ratio": 100.0}) == pytest.approx(0.4)


def test_large_trades_add_up_to_point_three(predictor):
    assert predictor._calculate_unusual_activity_score({"large_trade_count": 1}) == pytest.approx(0.1)
    assert predictor._calculate_unusual_activity_score({"large_trade_count": 50}) == pytest.approx(0.3)


def test_extreme_iv_rank_adds_point_two(predictor):
    assert predictor._calculate_unusual_activity_score({"iv_rank": 95}) == pytest.approx(0.2)
    assert predictor._calculate_unusual_activity_score({"iv_rank": 5}) == pytest.approx(0.2)
    assert predictor._calculate_unusual_activity_score({"iv_rank": 50}) == 0.0


def test_extreme_put_call_ratio_adds_point_one(predictor):
    assert predictor._calculate_unusual_activity_score({"put_call_ratio": 2.5}) == pytest.approx(0.1)
    assert predictor._calculate_unusual_activity_score({"put_call_ratio": 0.2}) == pytest.approx(0.1)


def test_unusual_score_never_exceeds_one(predictor):
    features = {"volume_ratio": 50, "large_trade_count": 99, "iv_rank": 99, "put_call_ratio": 5}
    assert predictor._calculate_unusual_activity_score(features) <= 1.0


@pytest.mark.parametrize(
    "unusual, confidence, expected",
    [(0.9, 0.1, "high"), (0.5, 0.5, "medium"), (0.0, 1.0, "low")],
)
def test_risk_level_combines_unusual_activity_and_doubt(predictor, unusual, confidence, expected):
    assert predictor._assess_risk_level(unusual, confidence) == expected


def test_itm_otm_ratio_is_one_without_strikes(predictor):
    assert predictor._calculate_itm_otm_ratio([], 100) == 1.0


def test_calls_below_the_price_are_in_the_money(predictor):
    strikes = [{"strike": 90, "type": "call"}, {"strike": 95, "type": "call"}, {"strike": 110, "type": "call"}]
    assert predictor._calculate_itm_otm_ratio(strikes, 100) == 2.0


def test_puts_above_the_price_are_in_the_money(predictor):
    strikes = [{"strike": 110, "type": "put"}, {"strike": 90, "type": "put"}]
    assert predictor._calculate_itm_otm_ratio(strikes, 100) == 1.0


def test_itm_otm_ratio_does_not_divide_by_zero(predictor):
    strikes = [{"strike": 90, "type": "call"}, {"strike": 91, "type": "call"}]
    assert predictor._calculate_itm_otm_ratio(strikes, 100) == 2.0


def test_days_to_expiry_defaults_to_thirty(predictor):
    assert predictor._calculate_avg_days_to_expiry({}) == 30.0
    assert predictor._calculate_avg_days_to_expiry({"expirations": []}) == 30.0


def test_days_to_expiry_ignores_bad_dates(predictor):
    assert predictor._calculate_avg_days_to_expiry({"expirations": [{"date": "not a date"}]}) == 30.0


def test_past_expirations_count_as_zero_days(predictor):
    result = predictor._calculate_avg_days_to_expiry({"expirations": [{"date": "2001-01-01"}]})
    assert result == 0


def test_default_features_are_neutral(predictor):
    features = predictor._get_default_features()
    assert features["put_call_ratio"] == 1.0
    assert features["call_volume_pct"] == features["put_volume_pct"] == 0.5
    assert features["iv_rank"] == 50.0


def test_feature_engineering_splits_volume(predictor):
    features = predictor._engineer_features({"call_volume": 300, "put_volume": 100})
    assert features["call_volume_pct"] == pytest.approx(0.75)
    assert features["put_volume_pct"] == pytest.approx(0.25)


def test_feature_engineering_defaults_without_volume(predictor):
    features = predictor._engineer_features({})
    assert features["call_volume_pct"] == features["put_volume_pct"] == 0.5
    assert features["volume_ratio"] == 1.0


def test_feature_engineering_splits_open_interest(predictor):
    features = predictor._engineer_features({"call_open_interest": 100, "put_open_interest": 300})
    assert features["call_oi_pct"] == pytest.approx(0.25)
    assert features["put_oi_pct"] == pytest.approx(0.75)


def test_feature_engineering_reads_iv_and_large_trades(predictor):
    features = predictor._engineer_features(
        {
            "implied_volatility": {"rank": 85, "percentile": 90, "skew": 0.2},
            "large_trades": [{"volume": 500}, {"volume": 700}],
        }
    )
    assert features["iv_rank"] == 85
    assert features["large_trade_count"] == 2
    assert features["large_trade_volume"] == 1200


def test_feature_engineering_reads_greeks(predictor):
    features = predictor._engineer_features({"greeks": {"delta": 0.4, "gamma": 0.02, "theta": -0.1, "vega": 0.3}})
    assert (features["net_delta"], features["net_gamma"]) == (0.4, 0.02)
    assert (features["net_theta"], features["net_vega"]) == (-0.1, 0.3)


def test_volume_ratio_compares_with_the_average(predictor):
    features = predictor._engineer_features({"call_volume": 300, "put_volume": 100, "avg_volume": 100})
    assert features["volume_ratio"] == pytest.approx(4.0)


def test_low_put_call_ratio_is_bullish(predictor):
    prediction = predictor._rule_based_prediction({"put_call_ratio": 0.5}, {})
    assert prediction.flow_sentiment == "bullish"
    assert "Low Put/Call ratio (bullish)" in prediction.key_indicators


def test_high_put_call_ratio_is_bearish(predictor):
    prediction = predictor._rule_based_prediction({"put_call_ratio": 1.6}, {})
    assert prediction.flow_sentiment == "bearish"


def test_balanced_flow_is_neutral(predictor):
    prediction = predictor._rule_based_prediction({"put_call_ratio": 1.0}, {})
    assert prediction.flow_sentiment == "neutral"
    assert prediction.confidence == 0.5


def test_confidence_is_capped_at_point_eight(predictor):
    features = {"put_call_ratio": 0.5, "volume_ratio": 5, "call_volume_pct": 0.9, "large_trade_count": 3, "iv_rank": 10}
    prediction = predictor._rule_based_prediction(features, {})
    assert prediction.flow_sentiment == "bullish"
    assert prediction.confidence == pytest.approx(0.8)


def test_large_trades_are_listed_as_an_indicator(predictor):
    prediction = predictor._rule_based_prediction({"large_trade_count": 2}, {})
    assert "Large trades detected (2)" in prediction.key_indicators


def test_at_most_five_indicators_are_returned(predictor):
    features = {"put_call_ratio": 0.5, "volume_ratio": 5, "iv_rank": 10, "large_trade_count": 3}
    assert len(predictor._rule_based_prediction(features, {}).key_indicators) <= 5


def test_fallback_prediction_is_cautious(predictor):
    prediction = predictor._fallback_prediction({})
    assert prediction.flow_sentiment == "neutral"
    assert prediction.confidence == 0.3
    assert prediction.key_indicators == ["fallback_analysis"]


def test_key_indicators_describe_high_volume(predictor):
    assert predictor._identify_key_indicators({"volume_ratio": 2.5}) == ["High volume (2.5x normal)"]
