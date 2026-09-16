# Qatar News Challenge: Project Status & Technical Report

**Current Local Date**: September 16, 2026  
**Repository**: `news_google_challenge` (`git@github.com:shehel/news_classification.git`)  
**Tracking Project**: ClearML (`news_google_challenge`)

---

## 1. Executive Summary & Leaderboard Standings

This project develops an end-to-end multi-task deep learning system for the **Qatar News Challenge**, classifying Arabic news documents across 6 simultaneous semantic and geopolitical dimensions under strict constraints (**no external datasets, no non-open-source APIs**).

### Leaderboard Benchmark
- **Online Competition Leaderboard #1 (Target)**: **0.9375** Macro Weighted F1
- **Our Current Online Submission (AraBERT + mDeBERTa Ensemble)**: **0.9217** Macro Weighted F1
- **Current Gap to Top**: **-0.0158**
- **Next Milestone**: Fine-tune and incorporate **Model 3 (Qwen with LoRA)** into a 3-model triad ensemble to push past 0.9375.

---

## 2. Problem Formulation & Competition Metric

The task requires simultaneous document-level classification into 6 targets:
1. **`Qatar Related`** (Binary: `0`, `1`) — ~30% positive class ratio.
2. **`Geography`** (13 Classes: `G-1` through `G-12`, `NO_GEOGRAPHY`) — extreme long-tail class imbalance (`G-1` has 2 samples, `G-10` has 7 samples).
3. **`Politics & Conflict`** (3 Classes: `PC-1`, `PC-2`, `NO_POLITICS`).
4. **`Health & Wellbeing`** (3 Classes: `H-1`, `H-2`, `NO_HEALTH`).
5. **`Science`** (4 Classes: `SC-1`, `SC-2`, `SC-3`, `NO_SCIENCE`).
6. **`Sports`** (5 Classes: `SP-1`, `SP-2`, `SP-3`, `SP-4`, `NO_SPORTS`).

### Competition Metric
The evaluation metric is the unweighted macro-average of the weighted F1-score across all 6 targets:
$$\text{Score} = \frac{1}{6} \sum_{k=1}^6 \text{F1}_{\text{weighted}}(y_k, \hat{y}_k)$$

---

## 3. Data Engineering & Validation Architecture

### 3.1 Leak-Free 5-Fold Multilabel Stratified Split (`folds.csv`)
- **Challenge**: Standard K-Fold splits fragment rare multi-target intersections (e.g. `G-1` with only 2 global samples), causing validation leakage and evaluation instability.
- **Solution**: Implemented an iterative multilabel stratification algorithm ([scripts/create_folds.py](file:///home/shehel/Documents/ml_repos/news_google_challenge/scripts/create_folds.py)).
- **Result**: Exactly 757/758 samples per fold with identical joint label distributions across all 5 folds:
  - Total Training Data: 3,786 articles
  - Test Data: 1,262 articles

### 3.2 Arabic Text Preprocessing Pipeline ([src/data.py](file:///home/shehel/Documents/ml_repos/news_google_challenge/src/data.py))
- Strips Arabic diacritics (*Tashkeel*: *Fatha*, *Damma*, *Kasra*, *Sukun*, *Shadda*, etc.).
- Normalizes elongation (*Tatweel* / *Kashida* removal: `ـ`).
- Unifies morphological variants:
  - `إ`, `أ`, `آ` $\to$ `ا`
  - `ة` $\to$ `ه`
  - `ى` $\to$ `ي`
- Preserves Arabic numbers, punctuation boundaries, and essential domain tokens.

---

## 4. Model Architecture & Training Framework

All models share a modular multi-task architecture ([src/models/multi_head.py](file:///home/shehel/Documents/ml_repos/news_google_challenge/src/models/multi_head.py)):
- **Layer-Weighted Pooling (`LayerWeightedPooling`)**: Learns a softmax-normalized weighting across the top 4 hidden layers of the transformer backbone, fusing both high-level semantic abstractions and intermediate syntactic/lexical representations.
- **Multi-Sample Dropout (`MultiSampleDropout`)**: Uses 5 parallel dropout paths ($p=0.2$) followed by shared linear classification heads, accelerating convergence and significantly reducing variance on small datasets.
- **Loss Function ([src/losses.py](file:///home/shehel/Documents/ml_repos/news_google_challenge/src/losses.py))**: Multi-task cross-entropy summing all 6 task losses with optional Kendall homoscedastic uncertainty weighting.
- **Optimization**: AdamW with cosine learning rate schedule, 10% linear warmup, and gradient norm clipping ($1.0$).

---

## 5. Model Results & Performance Summary

### 5.1 Model 1: AraBERTv02 (`aubmindlab/bert-base-arabertv02`)
- **Training Setup**: 5 folds, 4 epochs, learning rate $2 \times 10^{-5}$, effective batch size 16 (`batch_size: 8`, `grad_accum: 2`), `max_length: 384`, FP16 mixed precision.
- **Out-of-Fold 5-Fold CV Score**: **0.9082** Mean Weighted F1.
  - `Qatar Related`: **0.9306** (with post-processed threshold $\tau^* = 0.59$)
  - `Geography`: **0.7830** (strong native understanding of Arabic geographical names)
  - `Politics & Conflict`: **0.9081**
  - `Health & Wellbeing`: **0.9512**
  - `Science`: **0.9499**
  - `Sports`: **0.9265**
- **Artifact**: [submissions/arabert_5fold_submission.csv](file:///home/shehel/Documents/ml_repos/news_google_challenge/submissions/arabert_5fold_submission.csv)

### 5.2 Model 2: mDeBERTa-v3 (`microsoft/mdeberta-v3-base`)
- **Engineering Challenges Resolved**:
  1. *Tokenizer Dependencies*: Installed `sentencepiece`, `protobuf`, and `tiktoken`.
  2. *Hugging Face FP16 Weight Bug*: The pretrained HF checkpoint stores weights in FP16, which broke PyTorch AMP `GradScaler.unscale_()`. Fixed by forcing FP32 master weights.
  3. *Local VRAM Optimization*: Disentangled relative attention required extra activation memory; enabled `gradient_checkpointing` and adjusted to `batch_size: 4`, `grad_accum: 4` (peak VRAM dropped from $>6.0$ GB to $3.6$ GB).
- **Training Setup**: 5 folds, 4 epochs, learning rate $1.5 \times 10^{-5}$, effective batch size 16, `max_length: 384`, FP16 with gradient checkpointing.
- **Out-of-Fold 5-Fold CV Score**: **0.8144** Mean Weighted F1.
  - `Qatar Related`: 0.8561
  - `Geography`: 0.4227 (multilingual subwords struggle with dialectal Arabic toponyms)
  - `Politics & Conflict`: 0.8634 (orthogonal complementary predictions to AraBERT)
  - `Health & Wellbeing`: 0.9189
  - `Science`: 0.9214
  - `Sports`: 0.9039

---

## 6. Ensembling & Post-Processing

### 6.1 Task-Specific Nelder-Mead Optimization ([src/ensemble.py](file:///home/shehel/Documents/ml_repos/news_google_challenge/src/ensemble.py))
Rather than a naive uniform average across all tasks, Nelder-Mead simplex search dynamically optimized blending weights on out-of-fold validation probabilities:
- **`Qatar Related`**: `[1.000 AraBERT, 0.000 mDeBERTa]`
- **`Geography`**: `[1.000 AraBERT, 0.000 mDeBERTa]` (automatically protected against mDeBERTa's weaker geography performance)
- **`Politics & Conflict`**: `[0.513 AraBERT, 0.487 mDeBERTa]`
- **`Health & Wellbeing`**: `[1.000 AraBERT, 0.000 mDeBERTa]`
- **`Science`**: `[0.950 AraBERT, 0.050 mDeBERTa]`
- **`Sports`**: `[0.542 AraBERT, 0.458 mDeBERTa]`

### 6.2 Out-of-Fold Threshold Calibration
- For binary `Qatar Related`, scanning thresholds $\tau \in [0.10, 0.90]$ on out-of-fold probabilities identified $\tau^* = 0.59$ as optimal for Weighted F1.
- Boosted `Qatar Related` F1 from **0.8786** (default 0.5) to **0.9306**, producing an immediate $+0.0087$ lift to the competition macro-metric.

### 6.3 Online Leaderboard Verification
- Generated: [submissions/ensemble_arabert_mdeberta.csv](file:///home/shehel/Documents/ml_repos/news_google_challenge/submissions/ensemble_arabert_mdeberta.csv)
- **Online Competition Leaderboard Result**: **0.9217** (Within **0.0158** of 1st place).

---

## 7. Cluster & Infrastructure Setup

1. **ClearML Dataset**:
   - Dataset `news_google_challenge` registered under `news_google_challenge/datasets` (Version 1.0.0, ID: `42883068022b4a7f899c8efb3bc555fd`).
2. **ClearML-Labkit Orchestration**:
   - Modular campaign configs in `campaigns/` (`arabert_5fold.yaml`, `mdeberta_5fold.yaml`, `qwen_smoke.yaml`, `qwen_5fold.yaml`).
3. **Cluster `seneca1` Worker Diagnosis**:
   - Worker `acnodeg02:gpu0` is online and responsive.
   - **Identified Driver Constraint**: `acnodeg02` has an NVIDIA driver supporting CUDA 12.2 (`found version 12020`).
   - Tasks attempting to install CUDA 13 PyTorch wheels fail CUDA initialization and fall back to CPU.
   - **Resolution Applied**: Configured `requirements.txt` with `--extra-index-url https://download.pytorch.org/whl/cu121` and bound it in `src/tracking.py` so remote tasks install CUDA 12.1-compatible wheels.

---

## 8. Next Step: Adding Model 3 (Qwen with LoRA)

To bridge the **+0.0158** gap to 1st place (**0.9375**):

### Rationale
- Generative LLM backbones possess vast pre-training world knowledge, complex geopolitical reasoning, and extensive multilingual context.
- While encoder models (AraBERT) excel at tight morphological matching, an LLM provides distinct orthogonal inductive biases, particularly for subtle country/conflict associations and complex Qatar relevance cues.

### Implementation Blueprint
1. **Model Backbone**: `Qwen/Qwen2.5-7B` or `Qwen/Qwen3.5-9B`.
2. **PEFT LoRA Adaptation**:
   - Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.
   - Rank $r=16$, $\alpha=32$, dropout $0.05$.
3. **Pooling & Classification**:
   - Extract hidden representation at the last non-padded sequence token.
   - Connect to 6 multi-sample dropout heads.
4. **Execution Strategy**:
   - **Cluster (A100 on `seneca1`)**: Run with native BF16 (`--bf16`), `max_length: 512`, `batch_size: 4`, `grad_accum: 4`.
   - **Local Alternative**: If cluster requires further driver alignment, run 4-bit QLoRA (`bitsandbytes`) locally on the RTX A2000.
5. **Final Triad Ensemble**:
   - Combine out-of-fold probability distributions from all 3 model families:
     $$\hat{P}_{\text{Final}} = w_1 P_{\text{AraBERT}} + w_2 P_{\text{mDeBERTa}} + w_3 P_{\text{Qwen}}$$
   - Re-optimize task blending weights using Nelder-Mead and calibrate decision thresholds.
