"""Ensembling, Nelder-Mead task weight optimization, and submission generation."""
from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

from src.metrics import TARGET_COLUMNS, score_predictions


def blend_probabilities(
    probs_list: List[Dict[str, np.ndarray]],
    weights: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """Blends probability distributions across multiple models using task-specific weights."""
    blended: Dict[str, np.ndarray] = {}
    for col in TARGET_COLUMNS:
        w = weights[col]
        # Normalize weights
        w = w / np.sum(w)
        blended[col] = sum(w[m] * probs_list[m][col] for m in range(len(probs_list)))
    return blended


def optimize_task_weights(
    y_true: Dict[str, np.ndarray],
    oof_probs_list: List[Dict[str, np.ndarray]],
) -> Dict[str, np.ndarray]:
    """Finds optimal model weights per task to maximize weighted F1 on out-of-fold predictions."""
    num_models = len(oof_probs_list)
    optimal_weights: Dict[str, np.ndarray] = {}

    for col in TARGET_COLUMNS:
        true_labels = y_true[col]

        def loss_fn(raw_w: np.ndarray) -> float:
            w = np.clip(raw_w, 0, None)
            if np.sum(w) == 0:
                w = np.ones(num_models)
            w = w / np.sum(w)

            p_blend = sum(w[m] * oof_probs_list[m][col] for m in range(num_models))
            preds = np.argmax(p_blend, axis=-1)
            score = f1_score(true_labels, preds, average="weighted", zero_division=0)
            return -score

        init_w = np.ones(num_models) / num_models
        res = minimize(loss_fn, init_w, method="Nelder-Mead", options={"maxiter": 200})
        w_opt = np.clip(res.x, 0, None)
        w_opt = w_opt / np.sum(w_opt)
        optimal_weights[col] = w_opt
        print(f"Task '{col}' optimal weights: {np.round(w_opt, 4)} -> Best F1: {-res.fun:.4f}")

    return optimal_weights


def format_submission(
    test_ids: List[str],
    predicted_indices: Dict[str, np.ndarray],
    label_encoders: Dict[str, LabelEncoder],
    output_path: str = "submission.csv",
) -> pd.DataFrame:
    """Decodes integer class indices back to strings/binary values and writes the submission CSV."""
    df_sub = pd.DataFrame({"ID": test_ids})
    for col in TARGET_COLUMNS:
        indices = predicted_indices[col]
        if col == "Qatar Related":
            # Binary label (0 or 1)
            df_sub[col] = indices.astype(int)
        else:
            df_sub[col] = label_encoders[col].inverse_transform(indices)

    df_sub.to_csv(output_path, index=False)
    print(f"Saved submission with shape {df_sub.shape} to {output_path}")
    return df_sub
