import pandas as pd
import pytest

from modules.analytics import profile_dataset, clean_dataset, numeric_summary


def test_profile_dataset_counts_rows_columns_and_duplicates():
    df = pd.DataFrame({
        "name": ["Ali", "Sara", "Ali"],
        "score": [80, 90, 80],
    })
    result = profile_dataset(df)
    assert result["rows"] == 3
    assert result["columns"] == 2
    assert result["duplicate_rows"] == 1
    assert result["numeric_columns"] == ["score"]


def test_clean_dataset_removes_duplicate_rows():
    df = pd.DataFrame({
        "name": ["Ali", "Sara", "Ali"],
        "score": [80, 90, 80],
    })
    cleaned = clean_dataset(df)
    assert len(cleaned) == 2
    assert cleaned["name"].tolist() == ["Ali", "Sara"]


def test_clean_dataset_converts_blank_text_to_missing():
    df = pd.DataFrame({"name": ["Ali", "   "]})
    cleaned = clean_dataset(df)
    assert pd.isna(cleaned.loc[1, "name"])


def test_numeric_summary_returns_statistics():
    df = pd.DataFrame({"score": [60, 80, 100]})
    result = numeric_summary(df)
    assert result.loc["score", "mean"] == 80
    assert result.loc["score", "min"] == 60
    assert result.loc["score", "max"] == 100


def test_numeric_summary_returns_empty_dataframe_without_numbers():
    df = pd.DataFrame({"name": ["Ali", "Sara"]})
    result = numeric_summary(df)
    assert result.empty


def test_profile_dataset_rejects_non_dataframe():
    with pytest.raises(TypeError):
        profile_dataset([1, 2, 3])
