"""Convert loaded linguistic data into training-ready chat format.

Produces datasets in the Hugging Face `datasets` format with chat messages
compatible with the base model's chat template.
"""

import logging
import random
from pathlib import Path

from datasets import Dataset

from seshat.data.loaders import (
    DictionaryEntry,
    LanguageConfig,
    MonolingualText,
    ParallelSentence,
)

logger = logging.getLogger(__name__)


# --- Prompt Templates for Data Augmentation ---

DICTIONARY_PROMPTS = [
    "Translate the word '{headword}' to {bridge}.",
    "What does '{headword}' mean in {bridge}?",
    "How do you say '{headword}' in {bridge}?",
    "Provide the {bridge} translation of '{headword}'.",
    "Translate to {bridge}: {headword}",
]

DICTIONARY_REVERSE_PROMPTS = [
    "How do you say '{translation}' in {lang}?",
    "Translate '{translation}' from {bridge} to {lang}.",
    "What is the {lang} word for '{translation}'?",
    "Translate to {lang}: {translation}",
]

TRANSLATION_PROMPTS = [
    "Translate the following to {bridge}:\n{source}",
    "Translate to {bridge}:\n{source}",
    "What does this mean in {bridge}?\n{source}",
    "Provide a {bridge} translation:\n{source}",
]

TRANSLATION_REVERSE_PROMPTS = [
    "Translate the following to {lang}:\n{target}",
    "Translate to {lang}:\n{target}",
    "How would you say this in {lang}?\n{target}",
    "Provide a {lang} translation:\n{target}",
]

ANALYSIS_PROMPTS = [
    "Analyze and translate this {lang} sentence:\n{source}",
    "Break down the grammar and translate:\n{source}",
    "Provide a linguistic analysis of:\n{source}",
    "Translate and explain the structure of:\n{source}",
]


def _make_chat(
    system: str,
    user: str,
    assistant: str,
) -> dict:
    """Create a single chat training example in messages format."""
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


def format_dictionary_entries(
    entries: list[DictionaryEntry],
    lang_config: LanguageConfig,
    augment: bool = True,
    seed: int = 42,
) -> Dataset:
    """Convert dictionary entries to Phase 1 training examples.

    Args:
        entries: List of dictionary entries.
        lang_config: Language configuration.
        augment: Whether to create multiple prompt variations per entry.
        seed: Random seed for prompt selection.

    Returns:
        Hugging Face Dataset with chat-formatted training examples.
    """
    rng = random.Random(seed)
    examples: list[dict] = []
    system_prompt = lang_config.prompts.get(
        "translator",
        f"You are an expert translator for {lang_config.name}.",
    )

    for entry in entries:
        fmt_kwargs = {
            "headword": entry.headword,
            "translation": entry.translation,
            "lang": lang_config.name,
            "bridge": lang_config.bridge_language_name,
        }

        # Build assistant response
        response_parts = [f"{entry.headword} → {entry.translation}"]
        if entry.part_of_speech:
            response_parts.append(f"({entry.part_of_speech})")
        if entry.notes:
            response_parts.append(f"\nNote: {entry.notes}")
        assistant_response = " ".join(response_parts)

        if augment:
            # Forward translation (target lang → bridge)
            prompts_to_use = rng.sample(
                DICTIONARY_PROMPTS, min(3, len(DICTIONARY_PROMPTS))
            )
            for template in prompts_to_use:
                user_msg = template.format(**fmt_kwargs)
                examples.append(_make_chat(system_prompt, user_msg, assistant_response))

            # Reverse translation (bridge → target lang)
            reverse_response = f"{entry.translation} → {entry.headword}"
            if entry.part_of_speech:
                reverse_response += f" ({entry.part_of_speech})"

            reverse_prompts = rng.sample(
                DICTIONARY_REVERSE_PROMPTS,
                min(2, len(DICTIONARY_REVERSE_PROMPTS)),
            )
            for template in reverse_prompts:
                user_msg = template.format(**fmt_kwargs)
                examples.append(_make_chat(system_prompt, user_msg, reverse_response))
        else:
            user_msg = DICTIONARY_PROMPTS[0].format(**fmt_kwargs)
            examples.append(_make_chat(system_prompt, user_msg, assistant_response))

        # If entry has an example sentence, add it as a bonus pair
        if entry.example_source and entry.example_target:
            ex_user = f"Translate to {lang_config.bridge_language_name}:\n{entry.example_source}"
            examples.append(_make_chat(system_prompt, ex_user, entry.example_target))

    rng.shuffle(examples)
    logger.info(
        "Formatted %d training examples from %d dictionary entries",
        len(examples),
        len(entries),
    )
    return Dataset.from_list(examples)


def format_parallel_sentences(
    sentences: list[ParallelSentence],
    lang_config: LanguageConfig,
    augment: bool = True,
    seed: int = 42,
) -> Dataset:
    """Convert parallel sentences to Phase 2 training examples.

    Args:
        sentences: List of parallel sentence pairs.
        lang_config: Language configuration.
        augment: Whether to create bidirectional + multiple prompt variations.
        seed: Random seed.

    Returns:
        Hugging Face Dataset with chat-formatted training examples.
    """
    rng = random.Random(seed)
    examples: list[dict] = []
    system_prompt = lang_config.prompts.get(
        "translator",
        f"You are an expert translator for {lang_config.name}.",
    )

    for pair in sentences:
        fmt_kwargs = {
            "source": pair.source,
            "target": pair.target,
            "lang": lang_config.name,
            "bridge": lang_config.bridge_language_name,
        }

        if augment:
            # Forward: target language → bridge
            fwd_templates = rng.sample(
                TRANSLATION_PROMPTS, min(2, len(TRANSLATION_PROMPTS))
            )
            for template in fwd_templates:
                user_msg = template.format(**fmt_kwargs)
                examples.append(_make_chat(system_prompt, user_msg, pair.target))

            # Reverse: bridge → target language
            rev_template = rng.choice(TRANSLATION_REVERSE_PROMPTS)
            user_msg = rev_template.format(**fmt_kwargs)
            examples.append(_make_chat(system_prompt, user_msg, pair.source))
        else:
            user_msg = TRANSLATION_PROMPTS[0].format(**fmt_kwargs)
            examples.append(_make_chat(system_prompt, user_msg, pair.target))

    rng.shuffle(examples)
    logger.info(
        "Formatted %d training examples from %d parallel sentences",
        len(examples),
        len(sentences),
    )
    return Dataset.from_list(examples)


def format_analysis_examples(
    sentences: list[ParallelSentence],
    lang_config: LanguageConfig,
    analyses: list[str] | None = None,
    seed: int = 42,
) -> Dataset:
    """Convert parallel sentences with analysis annotations to Phase 3 training examples.

    If `analyses` is provided, each entry corresponds to a detailed morphological
    and grammatical analysis. Otherwise, a basic translation-with-explanation
    format is used.

    Args:
        sentences: List of parallel sentence pairs.
        lang_config: Language configuration.
        analyses: Optional list of analysis strings, one per sentence.
        seed: Random seed.

    Returns:
        Hugging Face Dataset with chat-formatted training examples.
    """
    rng = random.Random(seed)
    examples: list[dict] = []
    system_prompt = lang_config.prompts.get(
        "analyst",
        f"You are a linguist specializing in {lang_config.name}.",
    )

    for i, pair in enumerate(sentences):
        fmt_kwargs = {
            "source": pair.source,
            "lang": lang_config.name,
            "bridge": lang_config.bridge_language_name,
        }

        template = rng.choice(ANALYSIS_PROMPTS)
        user_msg = template.format(**fmt_kwargs)

        if analyses and i < len(analyses):
            assistant_response = analyses[i]
        else:
            # Fallback: basic translation response
            assistant_response = (
                f"Translation: {pair.target}\n\n"
                f"This sentence in {lang_config.name} translates to "
                f'"{pair.target}" in {lang_config.bridge_language_name}.'
            )

        examples.append(_make_chat(system_prompt, user_msg, assistant_response))

    rng.shuffle(examples)
    logger.info("Formatted %d analysis training examples", len(examples))
    return Dataset.from_list(examples)


def format_monolingual_texts(
    texts: list[MonolingualText],
    lang_config: LanguageConfig,
    seed: int = 42,
) -> Dataset:
    """Convert monolingual texts to Phase 4 generation training examples.

    These teach the model to produce text in the target language, using
    various generation prompts.

    Args:
        texts: List of monolingual text segments.
        lang_config: Language configuration.
        seed: Random seed.

    Returns:
        Hugging Face Dataset with chat-formatted training examples.
    """
    rng = random.Random(seed)
    examples: list[dict] = []
    system_prompt = lang_config.prompts.get(
        "generator",
        f"You are fluent in {lang_config.name}.",
    )

    generation_prompts = [
        f"Continue this text in {lang_config.name}:\n{{text_start}}",
        f"Write a sentence in {lang_config.name} about the following topic:\n{{topic}}",
        f"Express the following idea in {lang_config.name}:\n{{text_start}}",
    ]

    for text_item in texts:
        text = text_item.text

        # Use the text as a completion target with partial prompt
        words = text.split()
        if len(words) > 5:
            # Give first few words as prompt, rest as completion
            split_point = max(2, len(words) // 4)
            prompt_text = " ".join(words[:split_point])
            template = generation_prompts[0]
            user_msg = template.format(text_start=prompt_text)
            examples.append(_make_chat(system_prompt, user_msg, text))
        else:
            # Short text — use as-is
            user_msg = f"Write in {lang_config.name}: a short phrase or sentence."
            examples.append(_make_chat(system_prompt, user_msg, text))

    rng.shuffle(examples)
    logger.info("Formatted %d generation training examples", len(examples))
    return Dataset.from_list(examples)


def create_train_eval_split(
    dataset: Dataset,
    eval_ratio: float = 0.05,
    min_eval: int = 50,
    max_eval: int = 500,
    seed: int = 42,
) -> tuple[Dataset, Dataset]:
    """Split a dataset into training and evaluation sets.

    Args:
        dataset: Input dataset.
        eval_ratio: Fraction of data to use for evaluation.
        min_eval: Minimum number of evaluation examples.
        max_eval: Maximum number of evaluation examples.
        seed: Random seed.

    Returns:
        Tuple of (train_dataset, eval_dataset).
    """
    n_eval = int(len(dataset) * eval_ratio)
    n_eval = max(min_eval, min(n_eval, max_eval))
    n_eval = min(n_eval, len(dataset) - 1)  # Ensure at least 1 training example

    split = dataset.train_test_split(test_size=n_eval, seed=seed)
    logger.info(
        "Split: %d training, %d evaluation examples",
        len(split["train"]),
        len(split["test"]),
    )
    return split["train"], split["test"]


def save_dataset(dataset: Dataset, path: Path, name: str = "train") -> None:
    """Save a dataset to disk in Hugging Face format."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(path / name))
    logger.info("Saved dataset '%s' (%d examples) to %s", name, len(dataset), path)


def save_as_jsonl(dataset: Dataset, path: Path) -> None:
    """Save a dataset as JSONL for inspection or alternative tooling."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_json(str(path))
    logger.info("Saved %d examples as JSONL to %s", len(dataset), path)
