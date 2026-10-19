
import pandas as pd
from sklearn.ensemble import IsolationForest


def detect_anomalies(
    data: pd.DataFrame,
    features: list[str],
    contamination: float = 0.05,
) -> pd.DataFrame:
    """Detect unusual records and explain their strongest feature deviations."""

    if data.empty:
        raise ValueError("The dataset is empty.")

    if not 0.01 <= contamination <= 0.20:
        raise ValueError("Contamination must be between 0.01 and 0.20.")

    if not features:
        raise ValueError("Select at least one numerical feature.")

    if len(features) != len(set(features)):
        raise ValueError("Duplicate features are not allowed.")

    missing = [column for column in features if column not in data.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    selected = data[features].apply(pd.to_numeric, errors="coerce")

    if selected.isna().any().any():
        raise ValueError("Selected features contain missing or invalid values.")

    if not selected.map(lambda value: abs(value) != float("inf")).all().all():
        raise ValueError("Selected features must contain finite values.")

    if len(selected) < 5:
        raise ValueError("At least five records are required.")

    model = IsolationForest(
        n_estimators=150,
        contamination=contamination,
        random_state=42,
    )

    predictions = model.fit_predict(selected)
    scores = model.decision_function(selected)

    result = data.copy()
    result["anomaly_label"] = predictions
    result["AI result"] = [
        "Unusual" if prediction == -1 else "Typical"
        for prediction in predictions
    ]
    result["anomaly_score"] = scores

    # Use the median and interquartile range to compare feature values.
    medians = selected.median()
    scales = selected.quantile(0.75) - selected.quantile(0.25)

    # Fall back to standard deviation when the interquartile range is zero.
    standard_deviations = selected.std(ddof=0)
    scales = scales.where(scales > 0, standard_deviations)
    scales = scales.where(scales > 0, 1.0)

    explanations = []

    for position, prediction in enumerate(predictions):
        if prediction != -1:
            explanations.append("No strong anomaly detected by the model.")
            continue

        row = selected.iloc[position]
        deviations = ((row - medians) / scales).abs()
        strongest = deviations.sort_values(ascending=False).head(2)

        details = []

        for feature in strongest.index:
            value = row[feature]
            median = medians[feature]
            direction = "high" if value > median else "low"

            details.append(
                f"{feature} is {direction} "
                f"({value:g}; dataset median {median:g})"
            )

        explanations.append(
            "Unusual feature values: " + "; ".join(details) + "."
        )

    result["AI explanation"] = explanations

    return result
