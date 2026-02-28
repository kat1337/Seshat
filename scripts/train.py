"""Launch QLoRA fine-tuning for a target language."""

import logging
from pathlib import Path

import typer
from datasets import load_from_disk
from rich.console import Console

from seshat.training.trainer import SeshatTrainingConfig, train

app = typer.Typer(help="Train Seshat language model with QLoRA fine-tuning.")
console = Console()
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@app.command()
def main(
    config: Path = typer.Option("configs/base_qlora.yaml", "--config", help="Training config YAML"),
    lang: str = typer.Option(..., "--lang", help="Language code"),
    phase: int = typer.Option(1, "--phase", min=1, max=4, help="Training phase (1-4)"),
    model: str | None = typer.Option(None, "--model", help="Override base model (e.g., Qwen/Qwen3-8B)"),
    data_dir: Path = typer.Option("data/processed", "--data-dir", help="Processed data directory"),
    output_dir: str | None = typer.Option(None, "--output-dir", help="Override output directory"),
    resume: bool = typer.Option(False, "--resume", help="Resume from latest checkpoint"),
    epochs: int | None = typer.Option(None, "--epochs", help="Override number of epochs"),
    batch_size: int | None = typer.Option(None, "--batch-size", help="Override batch size"),
    lr: float | None = typer.Option(None, "--lr", help="Override learning rate"),
) -> None:
    """Run QLoRA fine-tuning for a specific training phase."""
    phase_names = {1: "phase1_vocabulary", 2: "phase2_translation", 3: "phase3_analysis", 4: "phase4_generation"}
    phase_name = phase_names[phase]

    console.print(f"[bold]Training Phase {phase}: {phase_name.split('_', 1)[1].title()}[/bold]")
    console.print(f"Language: {lang}")

    # Build overrides dict
    overrides: dict = {}
    if model:
        overrides["base_model"] = model
    if output_dir:
        overrides["output_dir"] = output_dir
    else:
        overrides["output_dir"] = f"outputs/{lang}/{phase_name}"
    if epochs:
        overrides["num_train_epochs"] = epochs
    if batch_size:
        overrides["per_device_train_batch_size"] = batch_size
    if lr:
        overrides["learning_rate"] = lr

    # Load config
    training_config = SeshatTrainingConfig.from_yaml(config, phase=phase, **overrides)
    console.print(f"Model: {training_config.base_model}")
    console.print(f"Output: {training_config.output_dir}")

    # Load datasets
    train_path = data_dir / lang / phase_name / "train"
    eval_path = data_dir / lang / phase_name / "eval"

    if not train_path.exists():
        console.print(f"[red]Training data not found: {train_path}[/red]")
        console.print("Run 'python scripts/prepare_data.py' first.")
        raise typer.Exit(1)

    train_dataset = load_from_disk(str(train_path))
    eval_dataset = load_from_disk(str(eval_path)) if eval_path.exists() else None

    console.print(f"Training examples: {len(train_dataset)}")
    if eval_dataset:
        console.print(f"Evaluation examples: {len(eval_dataset)}")

    # Train
    result_dir = train(
        training_config,
        train_dataset,
        eval_dataset,
        resume_from_checkpoint=resume,
    )

    console.print(f"\n[bold green]✓ Training complete![/bold green]")
    console.print(f"Output: {result_dir}")


if __name__ == "__main__":
    app()
