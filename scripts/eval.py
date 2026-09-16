"""Inference and evaluation script for trained checkpoints with ClearML artifact chaining."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tracking import (
    add_clearml_args,
    early_init,
    finalize_tracker,
    fetch_task_artifact,
    resolve_clearml_dataset,
    write_metrics_json,
)

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data import (
    NewsMultiTaskDataset,
    build_label_encoders,
    collate_multitask_fn,
)
from src.ensemble import format_submission
from src.evaluate import evaluate_model
from src.metrics import TARGET_COLUMNS
from src.models.arabert_model import build_arabert_model
from src.models.deberta_model import build_deberta_model
from src.models.qwen_model import build_qwen_model


def main():
    _clearml_task = early_init()

    parser = argparse.ArgumentParser(description="Evaluate checkpoint on test set or validation fold")
    add_clearml_args(parser)

    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--checkpoint_task_id", type=str, default=None, help="ClearML task ID to pull checkpoint from")
    parser.add_argument("--test_file", type=str, default="Test.csv")
    parser.add_argument("--folds_file", type=str, default="folds.csv")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--output_dir", type=str, default="experiments/eval")

    args = parser.parse_args()
    tracker = finalize_tracker(_clearml_task, args)

    resolve_clearml_dataset(args, path_fields=("test_file", "folds_file"))

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve checkpoint
    ckpt_file = args.checkpoint_path
    if not ckpt_file and args.checkpoint_task_id:
        # Check if local experiments directory already has best_model.pt
        parent_dir = str(Path(args.output_dir).parent)
        local_cand = os.path.join(parent_dir, "best_model.pt")
        if os.path.exists(local_cand):
            ckpt_file = local_cand
            print(f"[eval] Using existing local checkpoint: {ckpt_file}")
        else:
            try:
                print(f"[eval] Fetching checkpoint from ClearML task {args.checkpoint_task_id}...")
                ckpt_file = fetch_task_artifact(args.checkpoint_task_id, "best_checkpoint")
            except Exception as e:
                print(f"[eval] Warning fetching artifact: {e}")

    if not ckpt_file or not os.path.exists(ckpt_file):
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_file}")

    print(f"Loading checkpoint from: {ckpt_file}")
    checkpoint = torch.load(ckpt_file, map_location="cpu")
    train_args = checkpoint.get("args", {})
    model_type = train_args.get("model", "arabert")
    model_name = train_args.get("model_name")
    qatar_thresh = checkpoint.get("qatar_threshold", 0.5)

    df_folds = pd.read_csv(args.folds_file)
    encoders = build_label_encoders(df_folds)
    num_classes = {col: len(encoders[col].classes_) for col in TARGET_COLUMNS}

    if model_type == "arabert":
        m_name = model_name or "aubmindlab/bert-base-arabertv02"
        model, tokenizer = build_arabert_model(m_name, num_classes)
    elif model_type == "mdeberta":
        m_name = model_name or "microsoft/mdeberta-v3-base"
        model, tokenizer = build_deberta_model(m_name, num_classes)
    elif model_type == "qwen":
        m_name = model_name or "Qwen/Qwen3.5-9B"
        model, tokenizer = build_qwen_model(m_name, num_classes)
    else:
        raise ValueError(f"Unknown model: {model_type}")

    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.to(device)
    model.eval()

    test_df = pd.read_csv(args.test_file)
    print(f"Loaded {len(test_df)} test samples from {args.test_file}")

    test_dataset = NewsMultiTaskDataset(
        texts=test_df["text"].tolist(),
        tokenizer=tokenizer,
        max_length=args.max_length,
        ids=test_df["ID"].tolist(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_multitask_fn,
        num_workers=2,
    )

    _, test_probs, _ = evaluate_model(model, test_loader, device)

    # Save test probabilities
    out_probs_path = os.path.join(args.output_dir, "test_probs.npz")
    np.savez_compressed(out_probs_path, **test_probs)

    # Generate discrete predictions
    predicted_indices = {}
    for col in TARGET_COLUMNS:
        if col == "Qatar Related":
            pos_probs = test_probs[col][:, 1]
            predicted_indices[col] = (pos_probs >= qatar_thresh).astype(int)
        else:
            predicted_indices[col] = np.argmax(test_probs[col], axis=-1)

    sub_path = os.path.join(args.output_dir, "submission.csv")
    format_submission(
        test_ids=test_df["ID"].tolist(),
        predicted_indices=predicted_indices,
        label_encoders=encoders,
        output_path=sub_path,
    )

    if _clearml_task is not None:
        tracker.upload_artifact("predictions", out_probs_path)
        tracker.upload_artifact("submission_csv", sub_path)

    metrics_res = {"f1": float(checkpoint.get("mean_f1", 0.0))}
    write_metrics_json(args.output_dir, results=metrics_res, split="test", level="document", task=_clearml_task)
    tracker.close()


if __name__ == "__main__":
    main()
