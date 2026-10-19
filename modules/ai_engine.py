
import pandas as pd
from sklearn.ensemble import IsolationForest


def detect_anomalies(
    data: pd.DataFrame,
    features: list[str],
    contamination: float = 0.05,
) -> pd.DataFrame:
    """Return a copy of the data with model-generated anomaly labels."""

    if data.empty:
        raise ValueError("The dataset is empty.")

    if not 0.01 <= contamination <= 0.20:
        raise ValueError("Contamination must be between 0.01 and 0.20.")

    if not features:
        raise ValueError("Select at least one numerical feature.")

    missing = [column for column in features if column not in data.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    selected = data[features].apply(pd.to_numeric, errors="coerce")

    if selected.isna().any().any():
        raise ValueError("Selected features contain missing or invalid values.")

    if not selected.map(lambda value: pd.notna(value) and abs(value) != float("inf")).all().all():
        raise ValueError("Selected features must contain finite values.")

    if len(selected) < 5:
        raise ValueError("At least five records are required.")

    model = IsolationForest(
        n_estimators=150,
        contamination=contamination,
        random_state=42,
    )

    result = data.copy()
    predictions = model.fit_predict(selected)

    result["anomaly_label"] = predictions
    result["AI result"] = [
        "Unusual" if prediction == -1 else "Typical"
        for prediction in predictions
    ]

    return result
