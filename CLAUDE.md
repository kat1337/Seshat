# CLAUDE.md — Seshat: Low-Resource Language Model Fine-Tuning Toolkit

## Project Identity

**Seshat** is a toolkit for fine-tuning large multilingual language models (starting with Qwen 3 80B) to learn new languages — particularly endangered, indigenous, and low-resource languages. Named after the Egyptian goddess of writing and knowledge, Seshat aims to make LLM-based language preservation and linguistic research accessible to developers who aren't ML specialists.

The initial target is a South American Amazonic indigenous language, but the architecture MUST remain language-agnostic so it can be reused for any low-resource language pair.

## Hardware Context

Primary development and training hardware:
- **AMD AI MAX 395+ APU** with **128 GB unified memory** (CPU+GPU shared)
- This is an RDNA 3.5 iGPU on Strix Halo — NOT a datacenter GPU
- ROCm support is evolving for this chip. Always check compatibility before assuming GPU acceleration works
- 128 GB unified memory is shared between CPU and GPU — budget accordingly
- **Realistic training targets on this hardware**: QLoRA on models up to ~14B–32B parameters. For 80B, plan for cloud-assisted training or aggressive quantization (4-bit QLoRA with CPU offloading)
- **Inference**: Can run Qwen 3 80B quantized (Q4_K_M or similar) locally via llama.cpp or Ollama

## Architecture & Design Principles

1. **Pipeline-based**: Data flows through discrete stages: Ingest → Clean → Align → Format → Train → Evaluate
2. **Language-agnostic**: No hardcoded assumptions about the target language. Source language (for translations) defaults to Spanish but is configurable
3. **Config-driven**: All pipeline parameters, model choices, and training hyperparameters live in YAML config files, not hardcoded
4. **Reproducible**: Every training run is logged with full config snapshots. Use MLflow or Weights & Biases for experiment tracking
5. **Incremental**: Support adding new data and continuing training without starting over
6. **Accessible**: The primary user is a backend developer, NOT an ML engineer. Prefer high-level tools (Unsloth, Axolotl, LLaMA-Factory) over raw PyTorch training loops

## Tech Stack

### Core Training
- **Python 3.11+**
- **PyTorch 2.x** with ROCm backend (fallback to CPU if ROCm unavailable)
- **Hugging Face Transformers + PEFT** — LoRA/QLoRA adapter training
- **Unsloth** (preferred) — 2-5x faster LoRA fine-tuning, lower memory
- **Axolotl** (alternative) — YAML-config-based fine-tuning wrapper
- **BitsAndBytes** — 4-bit/8-bit quantization for QLoRA
- **llama.cpp / llama-cpp-python** — local inference with GGUF quantized models

### Data Processing
- **pandas** — tabular data manipulation
- **Whisper (openai-whisper or faster-whisper)** — audio transcription
- **Tesseract OCR + pytesseract** — scan/image text extraction
- **pdf2image + Pillow** — PDF scan processing
- **spaCy** — source-language NLP (tokenization, sentence splitting)
- **sentencepiece** — tokenizer analysis and potential vocabulary extension

### Evaluation & Tracking
- **sacrebleu** — BLEU score for translation evaluation
- **chrF** — character n-gram F-score (better for morphologically rich/agglutinative languages)
- **MLflow** or **wandb** — experiment tracking
- **Gradio** — web UI for testing and demo

### Infrastructure
- **Docker** — reproducible environment (especially for ROCm)
- **DVC** — data version control for large datasets
- **Make** or **Just** — task runner

## Project Structure

```
seshat/
├── CLAUDE.md
├── README.md
├── pyproject.toml
├── Makefile
├── docker/
│   ├── Dockerfile.train
│   ├── Dockerfile.inference
│   └── docker-compose.yml
├── configs/
│   ├── base.yaml                # Shared defaults
│   ├── training/
│   │   ├── qlora_qwen3_14b.yaml
│   │   ├── qlora_qwen3_80b.yaml
│   │   └── lora_defaults.yaml
│   ├── data/
│   │   └── pipeline.yaml
│   └── inference/
│       └── serving.yaml
├── src/seshat/
│   ├── data/
│   │   ├── ingest.py            # Raw data loading (PDFs, audio, text)
│   │   ├── ocr.py               # OCR pipeline for scanned documents
│   │   ├── transcribe.py        # Audio transcription via Whisper
│   │   ├── align.py             # Parallel sentence alignment
│   │   ├── clean.py             # Text normalization and cleaning
│   │   ├── format.py            # Convert to training formats
│   │   └── tokenizer.py         # Tokenizer analysis and extension
│   ├── training/
│   │   ├── train.py             # Main training entry point
│   │   ├── qlora.py             # QLoRA training setup
│   │   └── eval.py              # Evaluation metrics
│   ├── inference/
│   │   ├── serve.py             # Model serving
│   │   └── translate.py         # Translation interface
│   └── ui/
│       └── app.py               # Gradio demo
├── data/                        # .gitignored, tracked with DVC
│   ├── raw/
│   ├── processed/
│   └── training/
├── models/                      # .gitignored
├── notebooks/
├── tests/
├── scripts/
└── docs/
```

## Training Data Format

All training data → **chat/instruction JSONL** with mixed task types:

```jsonl
{"messages": [{"role": "system", "content": "You are a linguistic expert in {LANG}..."}, {"role": "user", "content": "Translate to Spanish: {L1_text}"}, {"role": "assistant", "content": "{ES_text}"}]}
{"messages": [{"role": "user", "content": "What does '{word}' mean in {LANG}?"}, {"role": "assistant", "content": "{definition_and_examples}"}]}
{"messages": [{"role": "user", "content": "Explain the grammar of: '{sentence}'"}, {"role": "assistant", "content": "{morphological_breakdown}"}]}
```

## Training Strategy

1. **Tokenizer check** — analyze Qwen 3's token fertility on target language. If >3x vs Spanish, extend vocabulary
2. **Start with Qwen 3 14B** — fits for QLoRA on 128GB. Prove the pipeline works
3. **QLoRA**: 4-bit NF4, LoRA rank 64-128, target all linear layers
4. **Mixed tasks**: translation both directions, dictionary, grammar, morphology
5. **Scale up** to 32B/80B once pipeline is validated
6. **Export**: merge adapter → quantize to GGUF → serve via llama.cpp

## Coding Conventions

- Python 3.11+, Ruff for formatting/linting
- Type hints required on all function signatures
- Google-style docstrings
- pytest for testing
- Pydantic for config validation, YAML for storage
- structlog for logging

## Key Commands

```bash
make install              # Install dependencies
make download-model       # Download base model
make ingest               # Process raw data
make ocr                  # OCR scanned documents
make transcribe           # Whisper on audio files
make align                # Align parallel sentences
make format               # Generate training JSONL
make train                # Run training (default config)
make eval                 # Evaluation suite
make serve                # Start inference server
make demo                 # Launch Gradio UI
```

## Critical Notes for AI Assistants

1. **Do NOT assume ROCm works on AI MAX 395+.** Always include CPU fallback paths.
2. **Memory is shared.** 128 GB is split between CPU and GPU. Always calculate memory budgets.
3. **User is a backend dev, not ML researcher.** Prefer high-level APIs. Explain ML concepts.
4. **Data quality > quantity** for low-resource languages. Prioritize cleaning over volume.
5. **Everything language-agnostic.** Use config variables, never hardcode language names.
6. **Respect the source material.** Indigenous language data may be culturally sensitive. Include access control config.
7. **Start small, iterate.** Prove pipeline on dictionary + transcriptions before OCR/audio.
8. **Speaker-aware translation** is a future goal. Design schemas to accommodate speaker/context metadata now.
9. **No formal grammar exists.** Model must learn implicitly. Include tasks that force grammatical generalization.
10. **End users are linguists.** They will ask about morphology, syntax, etymology. Evaluate accordingly.
