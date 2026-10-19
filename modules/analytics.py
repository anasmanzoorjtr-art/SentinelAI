
import pandas as pd


def profile_dataset(data: pd.DataFrame) -> dict:
    """Calculate basic data quality and structure metrics."""

    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    numeric_columns = data.select_dtypes(include="number").columns.tolist()
    categorical_columns = data.select_dtypes(include=["object", "str", "category", "bool"]).columns.tolist()


    return {
        "rows": int(len(data)),
        "columns": int(len(data.columns)),
        "duplicate_rows": int(data.duplicated().sum()),
        "missing_cells": int(data.isna().sum().sum()),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "column_names": data.columns.tolist(),
    }


def clean_dataset(data: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy without duplicate rows or blank-only text."""

    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    cleaned = data.copy()

    # Convert blank-only text cells to missing values.
    for column in cleaned.select_dtypes(include=["object", "string"]).columns:
        cleaned[column] = cleaned[column].replace(r"^\s*$", pd.NA, regex=True)

    # Remove exact duplicate rows.
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    return cleaned


def numeric_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive statistics for numerical columns."""

    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    numeric = data.select_dtypes(include="number")

    if numeric.empty:
        return pd.DataFrame()

    return numeric.describe().transpose()

import pandas as pd
import pytest

from modules.ai_engine import detect_anomalies


def test_detect_anomalies_adds_result_columns():
    df = pd.DataFrame({
        "requests": [10, 11, 12, 10, 11, 500],
        "risk_score": [1, 2, 2, 1, 2, 99],
    })
    result = detect_anomalies(df, ["requests", "risk_score"])

    assert "anomaly_label" in result.columns
    assert "AI result" in result.columns
    assert set(result["AI result"].unique()).issubset({"Typical", "Unusual"})


def test_detect_anomalies_identifies_extreme_record():
    df = pd.DataFrame({
        "requests": [10, 11, 12, 10, 11, 500],
        "risk_score": [1, 2, 2, 1, 2, 99],
    })
    result = detect_anomalies(df, ["requests", "risk_score"])

    assert result.iloc[-1]["AI result"] == "Unusual"


def test_detect_anomalies_preserves_original_data():
    df = pd.DataFrame({
        "requests": [10, 11, 12, 10, 11, 500],
        "risk_score": [1, 2, 2, 1, 2, 99],
    })
    original = df.copy()

    detect_anomalies(df, ["requests", "risk_score"])

    pd.testing.assert_frame_equal(df, original)


def test_detect_anomalies_rejects_empty_dataset():
    with pytest.raises(ValueError, match="empty"):
        detect_anomalies(pd.DataFrame(), ["requests"])


def test_detect_anomalies_rejects_invalid_contamination():
    df = pd.DataFrame({"requests": [1, 2, 3, 4, 5]})

    with pytest.raises(ValueError, match="Contamination"):
        detect_anomalies(df, ["requests"], contamination=0.5)


def test_detect_anomalies_rejects_missing_feature():
    df = pd.DataFrame({"requests": [1, 2, 3, 4, 5]})

    with pytest.raises(ValueError, match="Missing feature"):
        detect_anomalies(df, ["unknown_feature"])


def test_detect_anomalies_requires_five_records():
    df = pd.DataFrame({"requests": [1, 2, 3, 4]})

    with pytest.raises(ValueError, match="five records"):
        detect_anomalies(df, ["requests"])
