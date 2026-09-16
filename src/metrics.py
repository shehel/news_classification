"""Metric computation for news_google_challenge."""
from __future__ import annotations

from typing import Dict, Union
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

TARGET_COLUMNS = [
    "Qatar Related",
    "Geography",
    "Politics & Conflict",
    "Health & Wellbeing",
    "Science",
    "Sports",
]


def score_predictions(
    actual: Union[pd.DataFrame, Dict[str, np.ndarray]],
    predicted: Union[pd.DataFrame, Dict[str, np.ndarray]],
    target_columns: list[str] = TARGET_COLUMNS,
) -> Dict[str, float]:
    """Calculates weighted F1-score for each target column and the macro-average.

    Matches the exact evaluation metric defined in rules.md and starter notebooks.
    """
    scores: Dict[str, float] = {}
    for col in target_columns:
        y_true = actual[col]
        y_pred = predicted[col]
        if hasattr(y_true, "values"):
            y_true = y_true.values
        if hasattr(y_pred, "values"):
            y_pred = y_pred.values

        scores[col] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    scores["mean_f1"] = float(np.mean([scores[c] for c in target_columns]))
    return scores
