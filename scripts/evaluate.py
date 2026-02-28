"""Evaluate a fine-tuned model on held-out test data."""

import json
import logging
from pathlib import Path

import typer
from datasets import load_from_disk
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Evaluate a Seshat fine-tuned model.")
console = Console()
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@app.command()
def main(
    model_path: str = typer.Option(..., "--model", help="Path to model or adapter"),
    adapter_path: str | None = typer.Option(None, "--adapter", help="LoRA adapter path if separate"),
    lang: str = typer.Option(..., "--lang", help="Language code"),
    data_dir: Path = typer.Option("data/processed", "--data-dir", help="Processed data directory"),
    phase: int = typer.Option(2, "--phase", help="Phase to evaluate (uses eval split)"),
    metrics: str = typer.Option("bleu,chrf", "--metrics", help="Comma-separated metrics"),
    max_samples: int = typer.Option(200, "--max-samples", help="Max evaluation samples"),
    output_file: Path | None = typer.Option(None, "--output", help="Save results to JSON"),
) -> None:
    """Run evaluation metrics on a fine-tuned model."""
    import evaluate
    from seshat.inference.translator import SeshatTranslator

    phase_names = {1: "phase1_vocabulary", 2: "phase2_translation", 3: "phase3_analysis", 4: "phase4_generation"}
    phase_name = phase_names.get(phase, f"phase{phase}")

    eval_path = data_dir / lang / phase_name / "eval"
    if not eval_path.exists():
        console.print(f"[red]Eval data not found: {eval_path}[/red]")
        raise typer.Exit(1)

    eval_dataset = load_from_disk(str(eval_path))
    if len(eval_dataset) > max_samples:
        eval_dataset = eval_dataset.select(range(max_samples))

    console.print(f"[bold]Evaluating {model_path} on {len(eval_dataset)} examples[/bold]")

    # Load model
    translator = SeshatTranslator(
        model_path=model_path,
        adapter_path=adapter_path,
    )

    # Generate predictions
    predictions: list[str] = []
    references: list[str] = []

    for i, example in enumerate(eval_dataset):
        messages = example["messages"]
        # Extract user message and expected assistant response
        user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
        expected = next((m["content"] for m in messages if m["role"] == "assistant"), "")

        prediction = translator.generate(user_msg)
        predictions.append(prediction)
        references.append(expected)

        if (i + 1) % 50 == 0:
            console.print(f"  Processed {i + 1}/{len(eval_dataset)}")

    # Compute metrics
    metric_list = [m.strip() for m in metrics.split(",")]
    results: dict = {}

    if "bleu" in metric_list:
        bleu = evaluate.load("sacrebleu")
        bleu_result = bleu.compute(predictions=predictions, references=[[r] for r in references])
        results["bleu"] = round(bleu_result["score"], 2)

    if "chrf" in metric_list:
        chrf = evaluate.load("chrf")
        chrf_result = chrf.compute(predictions=predictions, references=[[r] for r in references])
        results["chrf"] = round(chrf_result["score"], 2)

    # Display results
    table = Table(title="Evaluation Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Score", style="green")
    for metric_name, score in results.items():
        table.add_row(metric_name.upper(), str(score))
    console.print(table)

    # Show some examples
    console.print("\n[bold]Sample predictions:[/bold]")
    for i in range(min(5, len(predictions))):
        console.print(f"[dim]Input:[/dim] {eval_dataset[i]['messages'][1]['content'][:100]}")
        console.print(f"[green]Expected:[/green] {references[i][:100]}")
        console.print(f"[blue]Predicted:[/blue] {predictions[i][:100]}")
        console.print()

    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        console.print(f"Results saved to {output_file}")


if __name__ == "__main__":
    app()
