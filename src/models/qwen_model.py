"""Model 3: Modern Generative LLM with LoRA Adaptation (Qwen/Qwen3.5-9B)."""
from __future__ import annotations

from typing import Dict, Optional
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

from src.metrics import TARGET_COLUMNS
from .multi_head import MultiSampleDropout

DEFAULT_QWEN_MODEL = "Qwen/Qwen3.5-9B"


class QwenMultiTaskClassifier(nn.Module):
    """Qwen LLM backbone with LoRA parameter-efficient adaptation and 6 multi-task heads."""

    def __init__(
        self,
        model_name_or_path: str = DEFAULT_QWEN_MODEL,
        num_classes: Dict[str, int] = None,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        dropout_rate: float = 0.2,
        load_in_4bit: bool = False,
    ):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        hidden_size = getattr(self.config, "hidden_size", None) or getattr(self.config, "d_model", 4096)
        self.hidden_size = hidden_size

        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        llm = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )

        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        self.llm = get_peft_model(llm, peft_config)

        self.heads = nn.ModuleDict()
        for col in TARGET_COLUMNS:
            n_out = num_classes[col]
            self.heads[col] = MultiSampleDropout(self.hidden_size, n_out, num_samples=5, drop_rate=dropout_rate)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        outputs = self.llm.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        last_hidden_state = outputs.last_hidden_state
        # Extract last non-padding token
        sequence_lengths = attention_mask.sum(dim=1) - 1
        sequence_lengths = torch.clamp(sequence_lengths, min=0)
        batch_size = input_ids.shape[0]
        pooled = last_hidden_state[torch.arange(batch_size, device=input_ids.device), sequence_lengths]

        logits = {col: head(pooled.float()) for col, head in self.heads.items()}
        return logits


def build_qwen_model(
    model_name: str = DEFAULT_QWEN_MODEL,
    num_classes: Dict[str, int] = None,
    lora_r: int = 16,
    lora_alpha: int = 32,
    dropout_rate: float = 0.2,
) -> tuple[QwenMultiTaskClassifier, AutoTokenizer]:
    """Instantiates Qwen model with LoRA and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = QwenMultiTaskClassifier(
        model_name_or_path=model_name,
        num_classes=num_classes,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        dropout_rate=dropout_rate,
    )
    return model, tokenizer
