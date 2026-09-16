"""Model 1: Arabic-Native Transformer Multi-Task Model (AraBERTv02 / MARBERTv2)."""
from __future__ import annotations

from typing import Dict
from transformers import AutoTokenizer

from .multi_head import MultiTaskEncoder

DEFAULT_ARABERT_MODEL = "aubmindlab/bert-base-arabertv02"


def build_arabert_model(
    model_name: str = DEFAULT_ARABERT_MODEL,
    num_classes: Dict[str, int] = None,
    dropout_rate: float = 0.2,
) -> tuple[MultiTaskEncoder, AutoTokenizer]:
    """Instantiates the AraBERT / MARBERT multi-task model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = MultiTaskEncoder(
        model_name_or_path=model_name,
        num_classes=num_classes,
        use_layer_pooling=True,
        use_msd=True,
        dropout_rate=dropout_rate,
    )
    return model, tokenizer
