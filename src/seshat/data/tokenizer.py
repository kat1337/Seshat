"""
Tokenizer analysis and compatibility checking for low-resource languages.

Analyzes how well a base model's tokenizer handles the target language.
High token fertility (many tokens per word) means the model has to work harder
to represent the language, which requires more training data and longer sequences.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import structlog

logger = structlog.get_logger()


@dataclass
class TokenizerReport:
    """Results of tokenizer analysis on a language sample."""

    model_name: str
    target_lang: str
    source_lang: str

    # Token fertility: avg tokens per word
    target_fertility: float
    source_fertility: float
    fertility_ratio: float  # target / source

    # Vocabulary coverage
    target_chars: set[str]
    unknown_chars: set[str]  # Characters not in tokenizer vocab
    char_coverage: float  # Fraction of chars that are known

    # Token statistics
    target_vocab_tokens: int  # Unique tokens used for target language
    avg_target_token_length: float

    # Recommendations
    recommendation: str

    def summary(self) -> str:
        lines = [
            f"=== Tokenizer Analysis: {self.model_name} ===",
            f"Target language: {self.target_lang}",
            f"Source language: {self.source_lang}",
            "",
            f"Token fertility (target): {self.target_fertility:.2f} tokens/word",
            f"Token fertility (source): {self.source_fertility:.2f} tokens/word",
            f"Fertility ratio: {self.fertility_ratio:.2f}x",
            "",
            f"Character coverage: {self.char_coverage:.1%}",
            f"Unknown characters: {self.unknown_chars or 'None'}",
            f"Unique tokens used: {self.target_vocab_tokens}",
            "",
            f"Recommendation: {self.recommendation}",
        ]
        return "\n".join(lines)


def analyze_tokenizer(
    model_name: str,
    target_texts: list[str],
    source_texts: list[str],
    target_lang: str = "target_language",
    source_lang: str = "Spanish",
) -> TokenizerReport:
    """Analyze a model's tokenizer compatibility with a target language.

    Args:
        model_name: HuggingFace model name (e.g., "Qwen/Qwen3-14B").
        target_texts: Sample texts in the target (indigenous) language.
        source_texts: Sample texts in the source (e.g., Spanish) language.
        target_lang: Name of the target language.
        source_lang: Name of the source language.

    Returns:
        TokenizerReport with analysis results and recommendations.
    """
    from transformers import AutoTokenizer

    logger.info("Loading tokenizer", model=model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Calculate token fertility
    target_fertility = _calculate_fertility(tokenizer, target_texts)
    source_fertility = _calculate_fertility(tokenizer, source_texts)
    fertility_ratio = target_fertility / source_fertility if source_fertility > 0 else float("inf")

    # Character coverage
    target_chars = set()
    for text in target_texts:
        target_chars.update(text)
    # Remove whitespace and common punctuation for analysis
    target_chars -= set(" \t\n\r.,;:!?\"'()-")

    unknown_chars = set()
    for char in target_chars:
        tokens = tokenizer.encode(char, add_special_tokens=False)
        # If a single character produces multiple tokens or the UNK token, it's unknown
        if len(tokens) > 2 or (len(tokens) == 1 and tokens[0] == tokenizer.unk_token_id):
            unknown_chars.add(char)

    char_coverage = 1.0 - (len(unknown_chars) / len(target_chars)) if target_chars else 1.0

    # Count unique tokens used
    all_target_tokens: Counter[int] = Counter()
    for text in target_texts:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        all_target_tokens.update(tokens)

    # Average token length in characters
    avg_token_length = 0.0
    if all_target_tokens:
        total_chars = sum(len(tokenizer.decode([t])) for t in all_target_tokens)
        avg_token_length = total_chars / len(all_target_tokens)

    # Generate recommendation
    if fertility_ratio < 1.5:
        recommendation = (
            "EXCELLENT: Tokenizer handles target language well. "
            "Proceed with standard fine-tuning."
        )
    elif fertility_ratio < 2.5:
        recommendation = (
            "ACCEPTABLE: Tokenizer is somewhat inefficient for target language. "
            "Fine-tuning will work but may need longer sequences and more data. "
            "Consider increasing max_seq_length in training config."
        )
    elif fertility_ratio < 4.0:
        recommendation = (
            "POOR: Tokenizer significantly over-tokenizes target language. "
            "Consider vocabulary extension with sentencepiece, or use a model "
            "with better multilingual tokenizer coverage. Training will require "
            "substantially more data for equivalent quality."
        )
    else:
        recommendation = (
            "CRITICAL: Tokenizer is extremely inefficient for target language. "
            "Vocabulary extension is strongly recommended before training. "
            "Without it, training will be very slow and results likely poor."
        )

    if unknown_chars:
        recommendation += (
            f"\nWARNING: {len(unknown_chars)} characters not in tokenizer vocabulary. "
            "These will be represented as byte-level tokens, which is functional "
            "but inefficient."
        )

    report = TokenizerReport(
        model_name=model_name,
        target_lang=target_lang,
        source_lang=source_lang,
        target_fertility=target_fertility,
        source_fertility=source_fertility,
        fertility_ratio=fertility_ratio,
        target_chars=target_chars,
        unknown_chars=unknown_chars,
        char_coverage=char_coverage,
        target_vocab_tokens=len(all_target_tokens),
        avg_target_token_length=avg_token_length,
        recommendation=recommendation,
    )

    logger.info("Tokenizer analysis complete", fertility_ratio=f"{fertility_ratio:.2f}")
    return report


def _calculate_fertility(tokenizer, texts: list[str]) -> float:
    """Calculate average tokens per whitespace-separated word."""
    total_tokens = 0
    total_words = 0

    for text in texts:
        words = text.split()
        tokens = tokenizer.encode(text, add_special_tokens=False)
        total_tokens += len(tokens)
        total_words += len(words)

    return total_tokens / total_words if total_words > 0 else 0.0


def load_sample_texts(filepath: str | Path, max_lines: int = 500) -> list[str]:
    """Load sample texts from a file (one text per line)."""
    texts = []
    with open(filepath, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            line = line.strip()
            if line:
                texts.append(line)
    return texts
