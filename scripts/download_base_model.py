#!/usr/bin/env python3
"""Download base model from HuggingFace Hub."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def download_model(config_path: str) -> None:
    """Download the base model specified in a training config."""
    from huggingface_hub import snapshot_download

    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_name = config["model"]["name"]
    revision = config["model"].get("revision", "main")
    output_dir = Path(config.get("paths", {}).get("models_base", "models/base"))
    output_dir.mkdir(parents=True, exist_ok=True)

    local_dir = output_dir / model_name.replace("/", "--")

    print(f"Downloading {model_name} (revision: {revision})")
    print(f"Destination: {local_dir}")

    snapshot_download(
        repo_id=model_name,
        revision=revision,
        local_dir=str(local_dir),
        ignore_patterns=["*.bin"],  # Prefer safetensors
    )

    print(f"✓ Model downloaded to {local_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download base model")
    parser.add_argument(
        "--config",
        default="configs/training/qlora_qwen3_14b.yaml",
        help="Training config file",
    )
    args = parser.parse_args()
    download_model(args.config)
