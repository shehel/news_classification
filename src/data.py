"""Data loading, text cleaning, dataset classes, and fold assignment."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

from .metrics import TARGET_COLUMNS

# Arabic text cleaning patterns
ARABIC_DIACRITICS = re.compile(r"[\u064B-\u0652\u0670\u0640]")
WHITESPACE = re.compile(r"\s+")
NON_TEXT = re.compile(r"[^\w\s\u0600-\u06FF]")


def clean_arabic_text(text: str) -> str:
    """Normalizes Arabic text by standardizing characters and removing diacritics."""
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    
    # Remove diacritics and tatweel
    text = ARABIC_DIACRITICS.sub("", text)
    
    # Normalize Alef variations
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ٱ", "ا")
    
    # Normalize Yaa / Alif Maqsura and Taa Marbuta
    text = text.replace("ى", "ي").replace("ة", "ه")
    
    # Clean whitespace and non-text characters
    text = NON_TEXT.sub(" ", text)
    text = WHITESPACE.sub(" ", text).strip()
    return text


def build_label_encoders(df: pd.DataFrame) -> Dict[str, LabelEncoder]:
    """Fits LabelEncoders on all target columns for multi-class indexing."""
    encoders = {}
    for col in TARGET_COLUMNS:
        le = LabelEncoder()
        le.fit(df[col].astype(str))
        encoders[col] = le
    return encoders


class NewsMultiTaskDataset(Dataset):
    """PyTorch Dataset for multi-task news document classification."""

    def __init__(
        self,
        texts: List[str],
        tokenizer: Any,
        max_length: int = 512,
        labels: Optional[Dict[str, np.ndarray]] = None,
        ids: Optional[List[str]] = None,
    ):
        self.texts = [clean_arabic_text(t) for t in texts]
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.labels = labels
        self.ids = ids

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }
        if "token_type_ids" in encoding:
            item["token_type_ids"] = encoding["token_type_ids"].squeeze(0)

        if self.labels is not None:
            item["labels"] = {
                col: torch.tensor(self.labels[col][idx], dtype=torch.long)
                for col in TARGET_COLUMNS
            }

        if self.ids is not None:
            item["id"] = self.ids[idx]

        return item


def collate_multitask_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function for multi-task batches."""
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])

    collated = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }

    if "token_type_ids" in batch[0]:
        collated["token_type_ids"] = torch.stack([item["token_type_ids"] for item in batch])

    if "labels" in batch[0]:
        collated["labels"] = {
            col: torch.stack([item["labels"][col] for item in batch])
            for col in TARGET_COLUMNS
        }

    if "id" in batch[0]:
        collated["ids"] = [item["id"] for item in batch]

    return collated
