# 𓋇 Seshat

**Low-resource language fine-tuning toolkit for large language models**

Seshat is a pipeline for teaching multilingual LLMs (Qwen 3, Llama, etc.) new languages using limited data — dictionaries, parallel texts, transcriptions, and audio recordings. Designed for language preservation, linguistic research, and translation of endangered and indigenous languages.

## Why Seshat?

Modern multilingual LLMs know 50-100+ languages but miss thousands of indigenous and endangered ones. Seshat bridges this gap by:

- **Ingesting** whatever data you have: dictionaries, bilingual texts, scans, audio recordings
- **Processing** it into clean, aligned training data
- **Fine-tuning** an existing multilingual model using LoRA/QLoRA (no ML PhD required)
- **Deploying** the result locally for translation, linguistic analysis, and research

## Quickstart

### Prerequisites
- Python 3.11+
- ~128 GB RAM recommended for training (works with less for smaller models)
- AMD ROCm or NVIDIA CUDA for GPU acceleration (CPU fallback available)

### Installation

```bash
git clone https://github.com/YOUR_USER/seshat.git
cd seshat
make install
```

### Prepare Your Data

Place your raw data in the `data/raw/` directory:
```
data/raw/
├── dictionary/          # Dictionary files (CSV, JSON, or text)
├── transcriptions/      # Transcribed texts (bilingual or monolingual)
├── translations/        # Parallel translated texts
├── scans/               # Scanned document images/PDFs
└── audio/               # Audio recordings (.wav, .mp3, .flac)
```

### Run the Pipeline

```bash
# 1. Process and clean raw data
make ingest

# 2. (Optional) OCR scanned documents
make ocr

# 3. (Optional) Transcribe audio
make transcribe

# 4. Align parallel sentences
make align

# 5. Generate training data
make format

# 6. Download base model and train
make download-model
make train

# 7. Evaluate
make eval

# 8. Launch demo UI
make demo
```

### Configuration

All configuration lives in `configs/`. Copy and modify for your language:

```bash
cp configs/training/qlora_qwen3_14b.yaml configs/training/my_language.yaml
# Edit my_language.yaml with your settings
make train CONFIG=configs/training/my_language.yaml
```

## Data Requirements

See [docs/DATA_REQUIREMENTS.md](docs/DATA_REQUIREMENTS.md) for detailed analysis.

**TL;DR** — For basic translation capability:

| What You Need | Minimum | Recommended |
|---|---|---|
| Parallel sentence pairs | 10,000 | 50,000+ |
| Dictionary entries | 2,000 | 10,000+ |
| Monolingual text (target lang) | 5,000 sentences | 30,000+ |

If you have ~1,000 pages of bilingual text, you likely have 30,000-60,000 sentence pairs — enough for a solid starting point.

## Hardware

Developed on AMD AI MAX 395+ (128 GB unified memory). Also works on:
- Any system with 32+ GB RAM (smaller models)
- NVIDIA GPUs with 24+ GB VRAM
- Cloud instances (AWS, GCP, Lambda Labs)

## Project Structure

```
seshat/
├── CLAUDE.md          # AI assistant instructions
├── configs/           # All configuration files
├── src/seshat/        # Source code
│   ├── data/          # Data processing pipeline
│   ├── training/      # Model training
│   ├── inference/     # Model serving
│   └── ui/            # Gradio demo
├── data/              # Data directory (gitignored, use DVC)
├── models/            # Model files (gitignored)
├── notebooks/         # Jupyter exploration notebooks
├── scripts/           # Utility scripts
├── tests/             # Test suite
└── docs/              # Documentation
```

## Acknowledgments

Named after **Seshat** (𓋇𓏏𓁐), the ancient Egyptian goddess of writing, wisdom, and knowledge.
