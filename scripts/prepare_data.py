"""Data preparation pipeline — process raw data into training-ready format."""

import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from seshat.data.augmenters import augment_with_context_variation, create_curriculum_subsets
from seshat.data.formatters import (
    create_train_eval_split,
    format_dictionary_entries,
    format_monolingual_texts,
    format_parallel_sentences,
    save_as_jsonl,
    save_dataset,
)
from seshat.data.loaders import LanguageConfig, load_all_data
from seshat.data.processors import (
    clean_dictionary_entries,
    clean_parallel_sentences,
    compute_data_statistics,
)

app = typer.Typer(help="Prepare linguistic data for Seshat training.")
console = Console()
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@app.command()
def main(
    lang: str = typer.Option(..., "--lang", help="Language code (matches config filename)"),
    phases: str = typer.Option("all", "--phases", help="Phases to prepare: all, dictionary, translation, analysis, generation"),
    data_root: Path = typer.Option("data/raw", "--data-root", help="Root directory for raw data"),
    output_root: Path = typer.Option("data/processed", "--output-root", help="Root directory for processed data"),
    config_dir: Path = typer.Option("configs/languages", "--config-dir", help="Directory containing language configs"),
    augment: bool = typer.Option(True, "--augment/--no-augment", help="Apply data augmentation"),
    seed: int = typer.Option(42, "--seed", help="Random seed for reproducibility"),
) -> None:
    """Process raw linguistic data into training-ready datasets."""
    # Load language config
    config_path = config_dir / f"{lang}.yaml"
    if not config_path.exists():
        console.print(f"[red]Language config not found: {config_path}[/red]")
        console.print("Copy configs/languages/template.yaml and fill in your language details.")
        raise typer.Exit(1)

    lang_config = LanguageConfig.from_yaml(config_path)
    console.print(f"[bold]Preparing data for: {lang_config.name} ({lang_config.code})[/bold]")

    # Load all raw data
    raw_data = load_all_data(lang_config, data_root)

    # Display statistics
    stats = compute_data_statistics(
        raw_data["dictionary"],
        raw_data["parallel"],
        raw_data["monolingual"],
    )

    table = Table(title="Raw Data Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    for key, value in stats.items():
        table.add_row(key.replace("_", " ").title(), str(value))
    console.print(table)

    phase_list = phases.split(",") if phases != "all" else ["dictionary", "translation", "analysis", "generation"]
    output_lang = output_root / lang

    # Phase 1: Dictionary
    if "dictionary" in phase_list and raw_data["dictionary"]:
        console.print("\n[bold cyan]Phase 1: Dictionary[/bold cyan]")
        entries = clean_dictionary_entries(raw_data["dictionary"])
        dataset = format_dictionary_entries(entries, lang_config, augment=augment, seed=seed)
        train_ds, eval_ds = create_train_eval_split(dataset, seed=seed)
        save_dataset(train_ds, output_lang / "phase1_vocabulary", "train")
        save_dataset(eval_ds, output_lang / "phase1_vocabulary", "eval")
        save_as_jsonl(train_ds, output_lang / "phase1_vocabulary" / "train.jsonl")
        console.print(f"  → {len(train_ds)} training, {len(eval_ds)} eval examples")

    # Phase 2: Translation
    if "translation" in phase_list and raw_data["parallel"]:
        console.print("\n[bold cyan]Phase 2: Translation[/bold cyan]")
        sentences = clean_parallel_sentences(raw_data["parallel"])
        if augment:
            sentences = augment_with_context_variation(
                sentences, lang_config.name, lang_config.bridge_language_name, seed=seed
            )
        dataset = format_parallel_sentences(sentences, lang_config, augment=augment, seed=seed)
        train_ds, eval_ds = create_train_eval_split(dataset, seed=seed)
        save_dataset(train_ds, output_lang / "phase2_translation", "train")
        save_dataset(eval_ds, output_lang / "phase2_translation", "eval")
        save_as_jsonl(train_ds, output_lang / "phase2_translation" / "train.jsonl")
        console.print(f"  → {len(train_ds)} training, {len(eval_ds)} eval examples")

    # Phase 3: Analysis (needs manual annotations — creates placeholder from parallel data)
    if "analysis" in phase_list and raw_data["parallel"]:
        console.print("\n[bold cyan]Phase 3: Analysis (basic — needs manual annotations)[/bold cyan]")
        from seshat.data.formatters import format_analysis_examples

        sentences = clean_parallel_sentences(raw_data["parallel"])
        # Use a subset for analysis (these should ideally be manually annotated)
        analysis_subset = sentences[:min(1000, len(sentences))]
        dataset = format_analysis_examples(analysis_subset, lang_config, seed=seed)
        train_ds, eval_ds = create_train_eval_split(dataset, seed=seed)
        save_dataset(train_ds, output_lang / "phase3_analysis", "train")
        save_dataset(eval_ds, output_lang / "phase3_analysis", "eval")
        console.print(f"  → {len(train_ds)} training, {len(eval_ds)} eval examples")
        console.print("  [yellow]⚠ These are basic examples. Add manual morphological annotations for better results.[/yellow]")

    # Phase 4: Generation
    if "generation" in phase_list and raw_data["monolingual"]:
        console.print("\n[bold cyan]Phase 4: Generation[/bold cyan]")
        dataset = format_monolingual_texts(raw_data["monolingual"], lang_config, seed=seed)
        train_ds, eval_ds = create_train_eval_split(dataset, seed=seed)
        save_dataset(train_ds, output_lang / "phase4_generation", "train")
        save_dataset(eval_ds, output_lang / "phase4_generation", "eval")
        console.print(f"  → {len(train_ds)} training, {len(eval_ds)} eval examples")

    console.print("\n[bold green]✓ Data preparation complete![/bold green]")
    console.print(f"Output directory: {output_lang}")


if __name__ == "__main__":
    app()
