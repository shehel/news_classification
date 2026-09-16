"""Model 2: Disentangled Multilingual Transformer Multi-Task Model (mDeBERTa-v3)."""
from __future__ import annotations

from typing import Dict
from transformers import AutoTokenizer

from .multi_head import MultiTaskEncoder

DEFAULT_MDEBERTA_MODEL = "microsoft/mdeberta-v3-base"


def build_deberta_model(
    model_name: str = DEFAULT_MDEBERTA_MODEL,
    num_classes: Dict[str, int] = None,
    dropout_rate: float = 0.2,
) -> tuple[MultiTaskEncoder, AutoTokenizer]:
    """Instantiates the mDeBERTa-v3 multi-task model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = MultiTaskEncoder(
        model_name_or_path=model_name,
        num_classes=num_classes,
        use_layer_pooling=True,
        use_msd=True,
        dropout_rate=dropout_rate,
    )
    return model, tokenizer
