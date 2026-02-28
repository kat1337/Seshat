# Training Guide

## Overview

This guide walks through fine-tuning a multilingual LLM to learn a new language using Seshat. The process has four phases: data preparation, tokenizer analysis, training, and evaluation.

## Prerequisites

```bash
make install
make download-model
```

## Phase 1: Data Preparation

### Step 1: Organize Raw Data

Place all source materials in the appropriate `data/raw/` subdirectories:

```
data/raw/
├── dictionary/          # Dictionary in CSV, JSON, or structured text
├── transcriptions/      # Bilingual transcriptions
├── translations/        # Parallel translated documents
├── scans/              # Scanned pages (PDF, PNG, JPG)
└── audio/              # Audio recordings
```

### Step 2: Configure the Pipeline

Edit `configs/data/pipeline.yaml` to match your data format:
- Set column names for your dictionary format
- Configure sentence alignment mode
- Set character whitelist for your target language
- Adjust task weights based on your use case priorities

### Step 3: Run the Pipeline

```bash
# Process structured data first (fastest)
make ingest

# Optional: process scans (can be slow, quality varies)
make ocr

# Optional: transcribe audio (requires Whisper, slow but high quality)
make transcribe

# Align parallel sentences
make align

# Generate training JSONL
make format

# Check data stats
make data-stats
```

### Step 4: Review Data Quality

Open `notebooks/01_data_exploration.ipynb` and check:
- Total number of training examples per task type
- Sentence length distributions
- Sample translations for correctness
- Any obvious encoding or alignment issues

## Phase 2: Tokenizer Analysis

Before training, check how well the base model's tokenizer handles your target language:

```bash
make tokenizer-analysis
```

This reports:
- **Token fertility**: Average tokens per word in your language vs Spanish
- **Unknown token rate**: How many characters fall outside the tokenizer's vocabulary
- **Character coverage**: Which characters in your language are handled natively

**Rules of thumb**:
- Token fertility < 2x Spanish: Tokenizer is fine, proceed
- Token fertility 2-3x Spanish: Acceptable but training will be slower
- Token fertility > 3x Spanish: Consider vocabulary extension (advanced, documented in `src/seshat/data/tokenizer.py`)

## Phase 3: Training

### Recommended Starting Config

```bash
# Start with 14B model (fast iteration, fits easily in 128 GB)
make train CONFIG=configs/training/qlora_qwen3_14b.yaml
```

### Monitor Training

If using MLflow:
```bash
mlflow ui --port 5000
# Open http://localhost:5000
```

Watch for:
- **Training loss** should decrease steadily
- **Eval loss** should decrease then plateau (not increase = overfitting)
- Training time per step on your hardware

### Common Issues

| Problem | Solution |
|---|---|
| OOM (Out of Memory) | Reduce `max_seq_length`, `per_device_train_batch_size`, or LoRA `r` |
| Loss not decreasing | Increase `learning_rate` by 2x, or check data quality |
| Loss spikes | Decrease `learning_rate`, increase `warmup_ratio` |
| Very slow training | Check if GPU is being used (`make check-gpu`), reduce model size |
| Garbage output | Data formatting issue — check JSONL structure matches chat template |

## Phase 4: Evaluation

```bash
make eval
```

This runs:
1. **BLEU** and **chrF** scores on the held-out test set
2. **Sample generations** for qualitative review
3. **Perplexity** on monolingual target language text

### Interpreting Results

**BLEU scores** (translation quality, 0-100):
- 10-20: Getting the gist, many errors
- 20-30: Understandable translations, noticeable errors
- 30-40: Good translations, minor errors
- 40+: Very good to excellent

**chrF scores** are generally more informative for morphologically rich languages and will be higher than BLEU.

### Qualitative Evaluation

The most important evaluation is manual review by language speakers or linguists. Check:
- Are translations accurate?
- Does the model use correct word forms/morphology?
- Can it handle idioms and culturally specific expressions?
- Are grammar explanations accurate?
- Does it hallucinate vocabulary that doesn't exist?

## Phase 5: Deployment

### Merge and Export

```bash
# Merge LoRA adapter into base model and quantize to GGUF
make export-gguf
```

### Run Locally

```bash
# Start inference server
make serve

# Or launch the web UI
make demo
```

### Share the Adapter

The LoRA adapter is small (typically 100-500 MB) and can be shared independently of the base model. Others can download the same base model and apply your adapter.

## Iterating

Fine-tuning is iterative. After your first run:

1. Review evaluation results and generated samples
2. Identify weak areas (vocabulary gaps, grammar errors, task types that underperform)
3. Add more data targeting weak areas
4. Adjust training config if needed (more epochs, different LR, higher LoRA rank)
5. Retrain and evaluate again

Each iteration should show improvement. If it doesn't, the bottleneck is usually data quality, not model size.
