"""Creates a leak-free, 5-fold multilabel stratified split for the challenge dataset."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.metrics import TARGET_COLUMNS


def iterative_stratified_split(Y: np.ndarray, n_splits: int = 5, seed: int = 42) -> np.ndarray:
    """Iterative stratification ensuring balanced multi-task label distributions."""
    np.random.seed(seed)
    n_samples, n_labels = Y.shape
    folds = [[] for _ in range(n_splits)]
    
    target_fold_size = n_samples / n_splits
    fold_label_counts = np.zeros((n_splits, n_labels))
    fold_sizes = np.zeros(n_splits)
    
    label_frequencies = Y.sum(axis=0)
    sample_rarity = Y.dot(1.0 / (label_frequencies + 1e-5))
    indices = np.argsort(-sample_rarity)
    
    for idx in indices:
        y = Y[idx]
        active_labels = np.where(y == 1)[0]
        
        best_fold = None
        best_score = float("inf")
        
        for f in range(n_splits):
            current_counts = fold_label_counts[f, active_labels]
            score = current_counts.sum() + 0.1 * (fold_sizes[f] / target_fold_size)
            if score < best_score:
                best_score = score
                best_fold = f
                
        folds[best_fold].append(idx)
        fold_sizes[best_fold] += 1
        fold_label_counts[best_fold] += y
        
    fold_assignments = np.zeros(n_samples, dtype=int)
    for f, idxs in enumerate(folds):
        fold_assignments[idxs] = f
    return fold_assignments


def main():
    parser = argparse.ArgumentParser(description="Generate CV folds")
    parser.add_argument("--train_file", type=str, default="Train.csv")
    parser.add_argument("--output_file", type=str, default="folds.csv")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.train_file)
    print(f"Loaded {len(df)} samples from {args.train_file}")

    one_hot_cols = []
    for t in TARGET_COLUMNS:
        dummies = pd.get_dummies(df[t], prefix=t)
        one_hot_cols.append(dummies)
    Y = pd.concat(one_hot_cols, axis=1).values.astype(int)

    folds = iterative_stratified_split(Y, n_splits=args.n_splits, seed=args.seed)
    df["fold"] = folds

    out_path = Path(args.output_file)
    df.to_csv(out_path, index=False)
    print(f"Saved folds to {out_path}")
    print("Fold counts:\n", df["fold"].value_counts().sort_index())


if __name__ == "__main__":
    main()
