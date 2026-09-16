"""Multi-task loss functions and weighting strategies."""
from __future__ import annotations

from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .metrics import TARGET_COLUMNS


class MultiTaskLoss(nn.Module):
    """Computes joint cross-entropy loss across all 6 targets with optional task weighting or uncertainty learning."""

    def __init__(
        self,
        class_weights: Optional[Dict[str, torch.Tensor]] = None,
        use_uncertainty_weighting: bool = False,
    ):
        super().__init__()
        self.class_weights = class_weights or {}
        self.use_uncertainty_weighting = use_uncertainty_weighting

        if use_uncertainty_weighting:
            # Learnable log variances for Kendall et al. uncertainty weighting
            self.log_vars = nn.ParameterDict({
                col: nn.Parameter(torch.zeros(1)) for col in TARGET_COLUMNS
            })

    def forward(
        self,
        logits_dict: Dict[str, torch.Tensor],
        labels_dict: Dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        total_loss = torch.tensor(0.0, device=next(iter(logits_dict.values())).device)
        losses_per_task: Dict[str, float] = {}

        for col in TARGET_COLUMNS:
            logits = logits_dict[col]
            labels = labels_dict[col]
            weights = self.class_weights.get(col)
            if weights is not None and weights.device != logits.device:
                weights = weights.to(logits.device)

            task_loss = F.cross_entropy(logits, labels, weight=weights)
            losses_per_task[col] = float(task_loss.detach().cpu().item())

            if self.use_uncertainty_weighting:
                precision = torch.exp(-self.log_vars[col])
                task_loss = precision * task_loss + 0.5 * self.log_vars[col]

            total_loss = total_loss + task_loss

        return total_loss, losses_per_task
