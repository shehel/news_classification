"""Evaluation loop, OOF prediction generation, and threshold optimization."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from src.metrics import TARGET_COLUMNS, score_predictions


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    loss_fn: Optional[nn.Module] = None,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray], Optional[Dict[str, np.ndarray]]]:
    """Runs evaluation over a DataLoader, returning metrics, predicted probabilities, and ground truth."""
    model.eval()
    all_probs: Dict[str, list] = {col: [] for col in TARGET_COLUMNS}
    all_targets: Dict[str, list] = {col: [] for col in TARGET_COLUMNS}
    total_val_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)

        logits_dict = model(input_ids, attention_mask, token_type_ids=token_type_ids)

        if "labels" in batch:
            labels_dict = {col: batch["labels"][col].to(device) for col in TARGET_COLUMNS}
            if loss_fn is not None:
                loss, _ = loss_fn(logits_dict, labels_dict)
                total_val_loss += loss.item()
                num_batches += 1

            for col in TARGET_COLUMNS:
                all_targets[col].extend(labels_dict[col].cpu().numpy().tolist())

        for col in TARGET_COLUMNS:
            probs = torch.softmax(logits_dict[col].float(), dim=-1).cpu().numpy()
            all_probs[col].append(probs)

    probs_dict = {col: np.concatenate(all_probs[col], axis=0) for col in TARGET_COLUMNS}

    if not all_targets[TARGET_COLUMNS[0]]:
        return {"val_loss": 0.0}, probs_dict, None

    targets_dict = {col: np.array(all_targets[col]) for col in TARGET_COLUMNS}

    # Argmax predictions for standard metrics
    preds_dict = {col: np.argmax(probs_dict[col], axis=-1) for col in TARGET_COLUMNS}
    metrics = score_predictions(targets_dict, preds_dict)
    metrics["val_loss"] = total_val_loss / max(num_batches, 1)

    return metrics, probs_dict, targets_dict


def optimize_qatar_threshold(
    y_true: np.ndarray,
    qatar_probs: np.ndarray,
) -> Tuple[float, float]:
    """Searches optimal decision threshold for Qatar Related binary classification to maximize weighted F1."""
    # qatar_probs shape: [N, 2], positive class is index 1
    pos_probs = qatar_probs[:, 1] if qatar_probs.ndim == 2 else qatar_probs
    best_thresh = 0.5
    best_f1 = -1.0

    for thresh in np.linspace(0.1, 0.9, 81):
        preds = (pos_probs >= thresh).astype(int)
        score = f1_score(y_true, preds, average="weighted", zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thresh = thresh

    return float(best_thresh), float(best_f1)
