
import pandas as pd
import pytest

from modules.ai_engine import detect_anomalies


def sample_data():
    return pd.DataFrame({
        "requests": [10, 11, 12, 10, 11, 500],
        "risk_score": [1, 2, 2, 1, 2, 99],
    })


def test_anomaly_detection_adds_result_columns():
    result = detect_anomalies(
        sample_data(), ["requests", "risk_score"]
    )
    assert "anomaly_label" in result.columns
    assert "AI result" in result.columns


def test_extreme_record_is_detected():
    result = detect_anomalies(
        sample_data(), ["requests", "risk_score"]
    )
    assert result.iloc[-1]["AI result"] == "Unusual"


def test_original_data_is_preserved():
    df = sample_data()
    original = df.copy()

    detect_anomalies(df, ["requests", "risk_score"])

    pd.testing.assert_frame_equal(df, original)


def test_empty_dataset_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        detect_anomalies(pd.DataFrame(), ["requests"])


def test_invalid_contamination_is_rejected():
    df = pd.DataFrame({"requests": [1, 2, 3, 4, 5]})

    with pytest.raises(ValueError, match="Contamination"):
        detect_anomalies(df, ["requests"], contamination=0.5)


def test_missing_feature_is_rejected():
    df = pd.DataFrame({"requests": [1, 2, 3, 4, 5]})

    with pytest.raises(ValueError, match="Missing feature"):
        detect_anomalies(df, ["unknown_feature"])


def test_fewer_than_five_records_are_rejected():
    df = pd.DataFrame({"requests": [1, 2, 3, 4]})

    with pytest.raises(ValueError, match="five records"):
        detect_anomalies(df, ["requests"])
