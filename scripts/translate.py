"""Interactive translation and analysis CLI."""

import logging
from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(help="Interactive translation with a fine-tuned Seshat model.")
console = Console()
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


@app.command()
def main(
    model_path: str = typer.Option(..., "--model", help="Path to merged model or base model"),
    adapter_path: str | None = typer.Option(None, "--adapter", help="Path to LoRA adapter (if not merged)"),
    lang_config: Path | None = typer.Option(None, "--lang-config", help="Language config YAML for system prompts"),
    load_in_4bit: bool = typer.Option(True, "--4bit/--no-4bit", help="Load model in 4-bit quantization"),
    interactive: bool = typer.Option(True, "--interactive/--no-interactive", help="Run interactive mode"),
    translate_text: str | None = typer.Option(None, "--text", help="Single text to translate (non-interactive)"),
) -> None:
    """Run translation using a fine-tuned Seshat model."""
    from seshat.inference.translator import SeshatTranslator

    # Load system prompt from language config if provided
    system_prompt = ""
    if lang_config and lang_config.exists():
        from seshat.data.loaders import LanguageConfig

        lc = LanguageConfig.from_yaml(lang_config)
        system_prompt = lc.prompts.get("translator", "")
        console.print(f"[bold]Language: {lc.name}[/bold]")

    translator = SeshatTranslator(
        model_path=model_path,
        adapter_path=adapter_path,
        load_in_4bit=load_in_4bit,
        system_prompt=system_prompt,
    )

    if translate_text:
        result = translator.generate(translate_text)
        console.print(result)
    elif interactive:
        translator.interactive()
    else:
        console.print("[yellow]Provide --text or use --interactive mode.[/yellow]")


if __name__ == "__main__":
    app()
