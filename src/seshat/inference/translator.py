"""Translation and analysis inference using fine-tuned models."""

import logging
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger(__name__)


class SeshatTranslator:
    """Interface for translation and linguistic analysis using a fine-tuned model.

    Can load either a merged model or a base model + LoRA adapter.
    """

    def __init__(
        self,
        model_path: str,
        adapter_path: str | None = None,
        load_in_4bit: bool = True,
        system_prompt: str = "",
        device_map: str = "auto",
    ) -> None:
        """Initialize the translator.

        Args:
            model_path: Path to merged model or base model.
            adapter_path: Path to LoRA adapter (if using base + adapter).
            load_in_4bit: Whether to quantize to 4-bit for inference.
            system_prompt: Default system prompt for conversations.
            device_map: Device placement strategy.
        """
        self.system_prompt = system_prompt

        logger.info("Loading model from %s", model_path)

        bnb_config = None
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            trust_remote_code=True,
        )

        if adapter_path:
            logger.info("Loading LoRA adapter from %s", adapter_path)
            self.model = PeftModel.from_pretrained(self.model, adapter_path)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.eval()
        logger.info("Model loaded and ready for inference.")

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_new_tokens: int = 512,
        temperature: float = 0.3,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> str:
        """Generate a response for a given prompt.

        Args:
            prompt: User message.
            system_prompt: Override default system prompt.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            do_sample: Whether to use sampling (vs greedy).

        Returns:
            Generated text response.
        """
        sys_prompt = system_prompt or self.system_prompt
        messages = []
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})
        messages.append({"role": "user", "content": prompt})

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else 1.0,
                top_p=top_p if do_sample else 1.0,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode only the generated tokens (not the prompt)
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(generated, skip_special_tokens=True)
        return response.strip()

    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        **kwargs,
    ) -> str:
        """Translate text between languages.

        Args:
            text: Text to translate.
            source_lang: Source language name.
            target_lang: Target language name.

        Returns:
            Translated text.
        """
        prompt = f"Translate the following from {source_lang} to {target_lang}:\n{text}"
        return self.generate(prompt, **kwargs)

    def analyze(self, text: str, lang_name: str, **kwargs) -> str:
        """Provide linguistic analysis of a text.

        Args:
            text: Text to analyze.
            lang_name: Name of the language.

        Returns:
            Analysis including translation, morphology, and grammar notes.
        """
        prompt = f"Analyze and translate this {lang_name} sentence:\n{text}"
        return self.generate(prompt, **kwargs)

    def interactive(self) -> None:
        """Run an interactive translation session in the terminal."""
        from rich.console import Console
        from rich.panel import Panel

        console = Console()
        console.print(Panel("Seshat Interactive Translator — type 'quit' to exit"))

        while True:
            try:
                user_input = console.input("[bold green]You:[/] ")
            except (EOFError, KeyboardInterrupt):
                break

            if user_input.lower() in ("quit", "exit", "q"):
                break

            response = self.generate(user_input)
            console.print(f"[bold blue]Seshat:[/] {response}\n")
