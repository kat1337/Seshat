.PHONY: install download-model ingest ocr transcribe align format train eval serve demo clean test lint

CONFIG ?= configs/training/qlora_qwen3_14b.yaml
DATA_CONFIG ?= configs/data/pipeline.yaml

# ─── Setup ────────────────────────────────────────────────────────────────────

install:
	pip install -e ".[all]"
	python -m spacy download es_core_news_sm

install-minimal:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

# ─── Data Pipeline ────────────────────────────────────────────────────────────

ingest:
	python -m seshat.data.ingest --config $(DATA_CONFIG)

ocr:
	python -m seshat.data.ocr --config $(DATA_CONFIG)

transcribe:
	python -m seshat.data.transcribe --config $(DATA_CONFIG)

align:
	python -m seshat.data.align --config $(DATA_CONFIG)

format:
	python -m seshat.data.format --config $(DATA_CONFIG)

pipeline: ingest align format
	@echo "✓ Full data pipeline complete"

# ─── Tokenizer ────────────────────────────────────────────────────────────────

tokenizer-analysis:
	python -m seshat.data.tokenizer --config $(DATA_CONFIG) --analyze

# ─── Training ─────────────────────────────────────────────────────────────────

download-model:
	python scripts/download_base_model.py --config $(CONFIG)

train:
	python -m seshat.training.train --config $(CONFIG)

eval:
	python -m seshat.training.eval --config $(CONFIG)

# ─── Inference & UI ───────────────────────────────────────────────────────────

serve:
	python -m seshat.inference.serve --config configs/inference/serving.yaml

demo:
	python -m seshat.ui.app

# ─── Export ───────────────────────────────────────────────────────────────────

export-gguf:
	bash scripts/export_gguf.sh

# ─── Dev ──────────────────────────────────────────────────────────────────────

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

format-code:
	ruff format src/ tests/

clean:
	rm -rf data/processed/* data/training/*
	rm -rf models/adapters/*
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .ipynb_checkpoints -exec rm -rf {} +

# ─── Info ─────────────────────────────────────────────────────────────────────

check-gpu:
	python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, ROCm: {torch.version.hip is not None if hasattr(torch.version, \"hip\") else False}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

data-stats:
	@echo "=== Data Statistics ==="
	@echo "Raw files:" && find data/raw -type f | wc -l
	@echo "Processed files:" && find data/processed -type f | wc -l 2>/dev/null || echo "0"
	@echo "Training examples:" && wc -l data/training/*.jsonl 2>/dev/null || echo "0"
