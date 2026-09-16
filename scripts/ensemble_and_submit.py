"""Aggregates OOF predictions from all models, optimizes task blending weights, and generates final submission."""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path
from typing import Dict, List

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.data import build_label_encoders
from src.ensemble import blend_probabilities, format_submission, optimize_task_weights
from src.evaluate import optimize_qatar_threshold
from src.metrics import TARGET_COLUMNS, score_predictions


def load_npz_dict(path: str) -> Dict[str, np.ndarray]:
    data = np.load(path)
    return {k: data[k] for k in data.files}


def main():
    parser = argparse.ArgumentParser(description="Ensemble OOF predictions and generate final submission")
    parser.add_argument("--experiments_dir", type=str, default="experiments")
    parser.add_argument("--folds_file", type=str, default="folds.csv")
    parser.add_argument("--test_file", type=str, default="Test.csv")
    parser.add_argument("--output_file", type=str, default="submissions/final_ensemble_submission.csv")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)

    df_folds = pd.read_csv(args.folds_file)
    encoders = build_label_encoders(df_folds)
    test_df = pd.read_csv(args.test_file)

    # Search for model experiment directories
    exp_dirs = [d for d in glob.glob(f"{args.experiments_dir}/*") if os.path.isdir(d)]
    print(f"Found {len(exp_dirs)} experiment directories in {args.experiments_dir}")

    # Discover models
    model_groups: Dict[str, list] = {}
    for d in exp_dirs:
        val_probs_path = os.path.join(d, "val_probs.npz")
        if os.path.exists(val_probs_path):
            name = os.path.basename(d)
            prefix = name.split("_fold")[0].split("_s")[0]
            model_groups.setdefault(prefix, []).append(d)

    print("Discovered model groups:")
    for m, paths in model_groups.items():
        print(f"  - {m}: {len(paths)} runs")

    if not model_groups:
        print("No valid experiment runs with 'val_probs.npz' found. Run training first.")
        sys.exit(1)

    # Aggregate OOF predictions per model family
    model_names = list(model_groups.keys())
    model_oof_probs: List[Dict[str, np.ndarray]] = []
    y_true: Dict[str, np.ndarray] = {}

    for m in model_names:
        paths = sorted(model_groups[m])
        combined_val_probs: Dict[str, list] = {col: [] for col in TARGET_COLUMNS}
        combined_val_targets: Dict[str, list] = {col: [] for col in TARGET_COLUMNS}

        for p in paths:
            val_p = load_npz_dict(os.path.join(p, "val_probs.npz"))
            val_t = load_npz_dict(os.path.join(p, "val_targets.npz"))
            for col in TARGET_COLUMNS:
                combined_val_probs[col].append(val_p[col])
                combined_val_targets[col].append(val_t[col])

        oof_dict = {col: np.concatenate(combined_val_probs[col], axis=0) for col in TARGET_COLUMNS}
        model_oof_probs.append(oof_dict)

        if not y_true:
            y_true = {col: np.concatenate(combined_val_targets[col], axis=0) for col in TARGET_COLUMNS}

        # Compute single model CV score
        preds = {col: np.argmax(oof_dict[col], axis=-1) for col in TARGET_COLUMNS}
        scores = score_predictions(y_true, preds)
        print(f"\n--- Model '{m}' Cross-Validation Scores ---")
        print(f"  Mean Weighted F1: {scores['mean_f1']:.4f}")
        for col in TARGET_COLUMNS:
            print(f"    {col:20}: {scores[col]:.4f}")

    # Optimize blending weights
    print("\n--- Optimizing Task-Specific Blending Weights ---")
    optimal_weights = optimize_task_weights(y_true, model_oof_probs)

    # Compute Ensembled OOF Predictions
    blended_oof = blend_probabilities(model_oof_probs, optimal_weights)
    qatar_opt_thresh, _ = optimize_qatar_threshold(y_true["Qatar Related"], blended_oof["Qatar Related"])

    blended_preds = {}
    for col in TARGET_COLUMNS:
        if col == "Qatar Related":
            pos_probs = blended_oof[col][:, 1]
            blended_preds[col] = (pos_probs >= qatar_opt_thresh).astype(int)
        else:
            blended_preds[col] = np.argmax(blended_oof[col], axis=-1)

    ensemble_scores = score_predictions(y_true, blended_preds)
    print("\n==============================================")
    print(f"=== ENSEMBLE CROSS-VALIDATION MEAN F1: {ensemble_scores['mean_f1']:.4f} ===")
    print("==============================================")
    for col in TARGET_COLUMNS:
        print(f"  {col:20}: {ensemble_scores[col]:.4f}")

    # Blend test predictions
    model_test_probs: List[Dict[str, np.ndarray]] = []
    has_test_preds = True

    for m in model_names:
        paths = sorted(model_groups[m])
        test_probs_for_model: Dict[str, list] = {col: [] for col in TARGET_COLUMNS}
        for p in paths:
            test_p_path = os.path.join(p, "test_probs.npz")
            if not os.path.exists(test_p_path):
                # Check eval subdirectory
                test_p_path = os.path.join(p, "eval", "test_probs.npz")
            if os.path.exists(test_p_path):
                tp = load_npz_dict(test_p_path)
                for col in TARGET_COLUMNS:
                    test_probs_for_model[col].append(tp[col])
            else:
                has_test_preds = False

        if test_probs_for_model[TARGET_COLUMNS[0]]:
            # Average across folds for this model
            avg_test = {
                col: np.mean(test_probs_for_model[col], axis=0) for col in TARGET_COLUMNS
            }
            model_test_probs.append(avg_test)

    if has_test_preds and len(model_test_probs) == len(model_names):
        blended_test = blend_probabilities(model_test_probs, optimal_weights)
        final_test_indices = {}
        for col in TARGET_COLUMNS:
            if col == "Qatar Related":
                pos_probs = blended_test[col][:, 1]
                final_test_indices[col] = (pos_probs >= qatar_opt_thresh).astype(int)
            else:
                final_test_indices[col] = np.argmax(blended_test[col], axis=-1)

        df_sub = format_submission(
            test_ids=test_df["ID"].tolist(),
            predicted_indices=final_test_indices,
            label_encoders=encoders,
            output_path=args.output_file,
        )
        print(f"\nFinal submission successfully written to {args.output_file}")
        print("Submission shape:", df_sub.shape)
        print("First 3 rows:\n", df_sub.head(3))
    else:
        print("\nNote: Test predictions not available for all models. Run eval.py to generate test predictions.")


if __name__ == "__main__":
    main()
