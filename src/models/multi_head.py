"""Multi-head classification components: Multi-Sample Dropout, Layer Pooling, and Head Wrappers."""
from __future__ import annotations

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from src.metrics import TARGET_COLUMNS


class MultiSampleDropout(nn.Module):
    """Multi-sample dropout for accelerated convergence and regularization."""

    def __init__(self, in_features: int, out_features: int, num_samples: int = 5, drop_rate: float = 0.2):
        super().__init__()
        self.dropouts = nn.ModuleList([nn.Dropout(drop_rate) for _ in range(num_samples)])
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Average logits over multiple dropout samples
        return torch.mean(torch.stack([self.linear(d(x)) for d in self.dropouts], dim=0), dim=0)


class LayerWeightedPooling(nn.Module):
    """Computes a learned weighted average of the last N hidden layers."""

    def __init__(self, num_layers: int = 4):
        super().__init__()
        self.num_layers = num_layers
        self.weights = nn.Parameter(torch.ones(num_layers) / num_layers)

    def forward(self, all_hidden_states: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        # Take last N layers
        selected = all_hidden_states[-self.num_layers:]
        stacked = torch.stack(selected, dim=0)  # [num_layers, batch_size, seq_len, hidden_dim]
        # Pool across sequence (mean pooling or CLS)
        cls_tokens = stacked[:, :, 0, :]  # [num_layers, batch_size, hidden_dim]
        norm_weights = torch.softmax(self.weights, dim=0).view(-1, 1, 1)
        pooled = torch.sum(cls_tokens * norm_weights, dim=0)
        return pooled


class MultiTaskEncoder(nn.Module):
    """Generic Multi-Task Transformer Encoder supporting AraBERT, MARBERT, mDeBERTa, etc."""

    def __init__(
        self,
        model_name_or_path: str,
        num_classes: Dict[str, int],
        use_layer_pooling: bool = True,
        use_msd: bool = True,
        dropout_rate: float = 0.2,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.config = AutoConfig.from_pretrained(
            model_name_or_path,
            output_hidden_states=True,
        )
        self.encoder = AutoModel.from_pretrained(model_name_or_path, config=self.config).float()
        if gradient_checkpointing and hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable()
        self.hidden_size = self.config.hidden_size
        self.use_layer_pooling = use_layer_pooling
        self.layer_pooler = LayerWeightedPooling(num_layers=4) if use_layer_pooling else None

        self.heads = nn.ModuleDict()
        for col in TARGET_COLUMNS:
            n_out = num_classes[col]
            if use_msd:
                self.heads[col] = MultiSampleDropout(self.hidden_size, n_out, num_samples=5, drop_rate=dropout_rate)
            else:
                self.heads[col] = nn.Sequential(
                    nn.Dropout(dropout_rate),
                    nn.Linear(self.hidden_size, n_out),
                )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids

        outputs = self.encoder(**kwargs)

        if self.use_layer_pooling and hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            pooled = self.layer_pooler(outputs.hidden_states)
        elif hasattr(outputs, "last_hidden_state"):
            # Mean pooling with attention mask
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
            sum_embeddings = torch.sum(outputs.last_hidden_state * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            pooled = sum_embeddings / sum_mask
        else:
            pooled = outputs[0][:, 0, :]

        logits = {col: head(pooled) for col, head in self.heads.items()}
        return logits
