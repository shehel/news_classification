"""Unified training entrypoint for news_google_challenge with ClearML and clearml-labkit support."""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tracking import (
    add_clearml_args,
    early_init,
    finalize_tracker,
    resolve_clearml_dataset,
    write_metrics_json,
)

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from src.data import (
    NewsMultiTaskDataset,
    build_label_encoders,
    collate_multitask_fn,
)
from src.evaluate import evaluate_model, optimize_qatar_threshold
from src.losses import MultiTaskLoss
from src.metrics import TARGET_COLUMNS
from src.models.arabert_model import build_arabert_model
from src.models.deberta_model import build_deberta_model
from src.models.qwen_model import build_qwen_model


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    # 1. EARLY INIT before any parse_args() runs
    _clearml_task = early_init()

    parser = argparse.ArgumentParser(description="Train multi-task Arabic news classifier")
    add_clearml_args(parser)

    # Model configuration
    parser.add_argument("--model", type=str, default="arabert", choices=["arabert", "mdeberta", "qwen"])
    parser.add_argument("--model_name", type=str, default=None, help="HuggingFace model name or path")
    parser.add_argument("--fold", type=int, default=0, help="Fold index to validate on (0-4)")
    parser.add_argument("--folds_file", type=str, default="folds.csv")
    parser.add_argument("--test_file", type=str, default="Test.csv")

    # Hyperparameters
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--use_uncertainty_weighting", action="store_true", default=False)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)

    # Outputs
    parser.add_argument("--output_dir", type=str, default="experiments/run")

    args = parser.parse_args()

    # 2. FINALIZE TRACKER right after parse_args()
    tracker = finalize_tracker(_clearml_task, args)

    # 3. Resolve ClearML dataset paths if applicable
    resolve_clearml_dataset(args, path_fields=("folds_file", "test_file"))

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Load data
    df = pd.read_csv(args.folds_file)
    encoders = build_label_encoders(df)
    num_classes = {col: len(encoders[col].classes_) for col in TARGET_COLUMNS}
    print(f"Target classes count: {num_classes}")

    # Build model & tokenizer
    if args.model == "arabert":
        m_name = args.model_name or "aubmindlab/bert-base-arabertv02"
        model, tokenizer = build_arabert_model(m_name, num_classes, dropout_rate=args.dropout)
    elif args.model == "mdeberta":
        m_name = args.model_name or "microsoft/mdeberta-v3-base"
        model, tokenizer = build_deberta_model(m_name, num_classes, dropout_rate=args.dropout)
    elif args.model == "qwen":
        m_name = args.model_name or "Qwen/Qwen3.5-9B"
        model, tokenizer = build_qwen_model(m_name, num_classes, dropout_rate=args.dropout)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    model.to(device)

    # Prepare splits
    train_df = df[df["fold"] != args.fold].reset_index(drop=True)
    val_df = df[df["fold"] == args.fold].reset_index(drop=True)
    print(f"Train samples: {len(train_df)}; Validation samples: {len(val_df)} (Fold {args.fold})")

    train_labels = {col: encoders[col].transform(train_df[col].astype(str)) for col in TARGET_COLUMNS}
    val_labels = {col: encoders[col].transform(val_df[col].astype(str)) for col in TARGET_COLUMNS}

    train_dataset = NewsMultiTaskDataset(
        texts=train_df["text"].tolist(),
        tokenizer=tokenizer,
        max_length=args.max_length,
        labels=train_labels,
        ids=train_df["ID"].tolist(),
    )
    val_dataset = NewsMultiTaskDataset(
        texts=val_df["text"].tolist(),
        tokenizer=tokenizer,
        max_length=args.max_length,
        labels=val_labels,
        ids=val_df["ID"].tolist(),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_multitask_fn,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        collate_fn=collate_multitask_fn,
        num_workers=2,
    )

    loss_fn = MultiTaskLoss(use_uncertainty_weighting=args.use_uncertainty_weighting).to(device)

    # Optimizer & Scheduler
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]
    if args.use_uncertainty_weighting:
        optimizer_grouped_parameters.append({"params": loss_fn.parameters(), "lr": args.lr})

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
    total_steps = (len(train_loader) // args.grad_accum) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    scaler = torch.amp.GradScaler("cuda", enabled=args.fp16 and torch.cuda.is_available())

    best_mean_f1 = -1.0
    best_metrics = {}
    best_probs = None

    print("\n--- Starting Training ---")
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader, 1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            labels_dict = {col: batch["labels"][col].to(device) for col in TARGET_COLUMNS}

            with torch.amp.autocast("cuda", enabled=args.fp16 and torch.cuda.is_available()):
                logits_dict = model(input_ids, attention_mask, token_type_ids=token_type_ids)
                loss, _ = loss_fn(logits_dict, labels_dict)
                if args.grad_accum > 1:
                    loss = loss / args.grad_accum

            scaler.scale(loss).backward()
            train_loss += loss.item() * args.grad_accum

            if step % args.grad_accum == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                scale_after = scaler.get_scale()
                # Step scheduler only if optimizer actually stepped (scale didn't decrease due to inf/nan)
                if scale_before <= scale_after:
                    scheduler.step()
                optimizer.zero_grad()

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        val_metrics, val_probs, val_targets = evaluate_model(model, val_loader, device, loss_fn=loss_fn)

        # Threshold optimization for Qatar Related
        qatar_true = val_targets["Qatar Related"]
        opt_thresh, opt_qatar_f1 = optimize_qatar_threshold(qatar_true, val_probs["Qatar Related"])

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_metrics['val_loss']:.4f} | Val Mean F1: {val_metrics['mean_f1']:.4f} "
            f"(Qatar Opt Thresh {opt_thresh:.2f}: {opt_qatar_f1:.4f})"
        )

        # Report to tracker
        tracker.report_scalar("Loss", "Train", avg_train_loss, iteration=epoch)
        tracker.report_scalar("Loss", "Val", val_metrics["val_loss"], iteration=epoch)
        tracker.report_scalar("Metrics", "Mean_F1", val_metrics["mean_f1"], iteration=epoch)
        for col in TARGET_COLUMNS:
            tracker.report_scalar("Task_F1", col, val_metrics[col], iteration=epoch)

        if val_metrics["mean_f1"] > best_mean_f1:
            best_mean_f1 = val_metrics["mean_f1"]
            best_metrics = val_metrics.copy()
            best_probs = val_probs

            # Save checkpoint
            ckpt_path = os.path.join(args.output_dir, "best_model.pt")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "mean_f1": best_mean_f1,
                    "metrics": best_metrics,
                    "args": vars(args),
                    "qatar_threshold": opt_thresh,
                },
                ckpt_path,
            )
            # Save OOF validation probabilities
            np.savez_compressed(os.path.join(args.output_dir, "val_probs.npz"), **val_probs)
            np.savez_compressed(os.path.join(args.output_dir, "val_targets.npz"), **val_targets)

    print(f"\n--- Best Validation Mean F1: {best_mean_f1:.4f} ---")
    for col in TARGET_COLUMNS:
        print(f"  {col:20}: {best_metrics[col]:.4f}")

    # Write metrics.json adhering to clearml-labkit registry contract
    metrics_json_path = write_metrics_json(
        output_dir=args.output_dir,
        results=best_metrics,
        split="val",
        level="document",
        task=_clearml_task,
    )
    print(f"Wrote metrics summary to: {metrics_json_path}")

    # Upload artifacts to ClearML
    if _clearml_task is not None:
        ckpt_path = os.path.join(args.output_dir, "best_model.pt")
        if os.path.exists(ckpt_path):
            tracker.upload_artifact("best_checkpoint", ckpt_path)
        val_probs_path = os.path.join(args.output_dir, "val_probs.npz")
        if os.path.exists(val_probs_path):
            tracker.upload_artifact("predictions", val_probs_path)

    tracker.close()


if __name__ == "__main__":
    main()
