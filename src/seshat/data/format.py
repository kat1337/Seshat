"""
Convert processed data into training-ready JSONL format.

This module takes cleaned, aligned parallel corpus data and dictionary entries
and produces instruction-tuning formatted JSONL files for LoRA/QLoRA training.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger()


@dataclass
class TrainingExample:
    """A single training example in chat/instruction format."""

    messages: list[dict[str, str]]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"messages": self.messages, "metadata": self.metadata}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


class TrainingDataFormatter:
    """Generates training examples from processed linguistic data.

    This class is language-agnostic. Language-specific details come from config.

    Args:
        lang_name: Human-readable name of the target language.
        source_lang_name: Human-readable name of the source/reference language.
        system_prompts: Dict mapping task types to system prompt templates.
            Use {LANG} as placeholder for lang_name.
        task_weights: Dict mapping task types to sampling weights.
        seed: Random seed for reproducibility.
    """

    TASK_TYPES = [
        "translation_to_source",
        "translation_to_target",
        "dictionary_lookup",
        "grammar_explanation",
        "morphological_analysis",
        "monolingual_completion",
    ]

    def __init__(
        self,
        lang_name: str,
        source_lang_name: str = "Spanish",
        system_prompts: dict[str, str] | None = None,
        task_weights: dict[str, float] | None = None,
        seed: int = 42,
    ) -> None:
        self.lang_name = lang_name
        self.source_lang_name = source_lang_name
        self.seed = seed
        self.rng = random.Random(seed)

        # Default system prompts
        self.system_prompts = system_prompts or {
            "translation": (
                f"You are a linguistic expert and translator for {lang_name}. "
                "Provide accurate translations and explain notable linguistic "
                "features when relevant."
            ),
            "dictionary": (
                f"You are a lexicographer specializing in {lang_name}. "
                "Provide detailed word definitions, usage examples, and "
                "etymological notes when available."
            ),
            "grammar": (
                f"You are a descriptive linguist studying {lang_name}. "
                "Explain grammatical structures, morphological patterns, "
                "and syntactic rules based on observed language data."
            ),
        }

        self.task_weights = task_weights or {
            "translation_to_source": 0.30,
            "translation_to_target": 0.30,
            "dictionary_lookup": 0.15,
            "grammar_explanation": 0.10,
            "morphological_analysis": 0.05,
            "monolingual_completion": 0.10,
        }

    def from_parallel_pair(
        self, source_text: str, target_text: str, metadata: dict | None = None
    ) -> list[TrainingExample]:
        """Generate training examples from a parallel sentence pair.

        Produces both translation directions from a single pair.
        """
        examples = []
        meta = metadata or {}

        # L1 → Source language (e.g., Indigenous → Spanish)
        examples.append(
            TrainingExample(
                messages=[
                    {"role": "system", "content": self.system_prompts["translation"]},
                    {
                        "role": "user",
                        "content": f"Translate to {self.source_lang_name}: {source_text}",
                    },
                    {"role": "assistant", "content": target_text},
                ],
                metadata={**meta, "task": "translation_to_source"},
            )
        )

        # Source language → L1 (e.g., Spanish → Indigenous)
        examples.append(
            TrainingExample(
                messages=[
                    {"role": "system", "content": self.system_prompts["translation"]},
                    {
                        "role": "user",
                        "content": f"Translate to {self.lang_name}: {target_text}",
                    },
                    {"role": "assistant", "content": source_text},
                ],
                metadata={**meta, "task": "translation_to_target"},
            )
        )

        return examples

    def from_dictionary_entry(
        self,
        headword: str,
        definition: str,
        pos: str | None = None,
        examples: list[str] | None = None,
        notes: str | None = None,
    ) -> list[TrainingExample]:
        """Generate training examples from a dictionary entry."""
        training_examples = []

        # Word meaning lookup
        response_parts = [definition]
        if pos:
            response_parts.insert(0, f"({pos})")
        if examples:
            response_parts.append("Examples: " + "; ".join(examples))
        if notes:
            response_parts.append(f"Note: {notes}")

        response = " ".join(response_parts)

        training_examples.append(
            TrainingExample(
                messages=[
                    {"role": "system", "content": self.system_prompts["dictionary"]},
                    {
                        "role": "user",
                        "content": f"What does '{headword}' mean in {self.lang_name}?",
                    },
                    {"role": "assistant", "content": response},
                ],
                metadata={"task": "dictionary_lookup", "headword": headword},
            )
        )

        # Reverse lookup (definition → word)
        training_examples.append(
            TrainingExample(
                messages=[
                    {"role": "system", "content": self.system_prompts["dictionary"]},
                    {
                        "role": "user",
                        "content": (
                            f"What is the word for '{definition}' in {self.lang_name}?"
                        ),
                    },
                    {"role": "assistant", "content": f"The word is '{headword}'."},
                ],
                metadata={"task": "dictionary_lookup_reverse", "headword": headword},
            )
        )

        return training_examples

    def from_grammar_example(
        self, sentence: str, explanation: str, metadata: dict | None = None
    ) -> TrainingExample:
        """Generate a grammar explanation training example."""
        return TrainingExample(
            messages=[
                {"role": "system", "content": self.system_prompts["grammar"]},
                {
                    "role": "user",
                    "content": f"Explain the grammatical structure of: '{sentence}'",
                },
                {"role": "assistant", "content": explanation},
            ],
            metadata={**(metadata or {}), "task": "grammar_explanation"},
        )

    def write_splits(
        self,
        examples: list[TrainingExample],
        output_dir: str | Path,
        train_ratio: float = 0.9,
        val_ratio: float = 0.05,
        test_ratio: float = 0.05,
    ) -> dict[str, int]:
        """Shuffle and split examples into train/val/test JSONL files.

        Returns dict with counts per split.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Shuffle
        shuffled = list(examples)
        self.rng.shuffle(shuffled)

        # Split
        n = len(shuffled)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        splits = {
            "train": shuffled[:train_end],
            "val": shuffled[train_end:val_end],
            "test": shuffled[val_end:],
        }

        counts = {}
        for split_name, split_examples in splits.items():
            filepath = output_dir / f"{split_name}.jsonl"
            with open(filepath, "w", encoding="utf-8") as f:
                for example in split_examples:
                    f.write(example.to_json() + "\n")
            counts[split_name] = len(split_examples)
            logger.info(
                "Wrote split",
                split=split_name,
                count=len(split_examples),
                path=str(filepath),
            )

        return counts
