"""Data augmentation strategies for low-resource language training.

These techniques help maximize the value of limited linguistic data
by creating valid training variations without corrupting meaning.
"""

import logging
import random

from seshat.data.loaders import DictionaryEntry, ParallelSentence

logger = logging.getLogger(__name__)


def augment_with_context_variation(
    sentences: list[ParallelSentence],
    lang_name: str,
    bridge_name: str,
    multiplier: int = 3,
    seed: int = 42,
) -> list[ParallelSentence]:
    """Create variations by changing the instruction context around the same content.

    This doesn't change the actual translation data, only the framing,
    teaching the model to respond to different phrasings of the same request.
    """
    rng = random.Random(seed)
    augmented: list[ParallelSentence] = list(sentences)  # Keep originals

    context_templates = [
        "Please translate this {lang} text: {src}",
        "{src}",  # Raw, no instruction
        "I need a {bridge} translation of: {src}",
        "What's the {bridge} equivalent of: {src}",
        "Can you translate this from {lang}? {src}",
    ]

    for pair in sentences:
        templates = rng.sample(context_templates, min(multiplier, len(context_templates)))
        for template in templates:
            new_source = template.format(
                src=pair.source,
                lang=lang_name,
                bridge=bridge_name,
            )
            augmented.append(
                ParallelSentence(
                    source=new_source,
                    target=pair.target,
                    domain=pair.domain,
                    source_file=pair.source_file,
                    confidence=pair.confidence * 0.95,  # Slightly lower confidence
                )
            )

    logger.info(
        "Augmented %d → %d sentences with context variation",
        len(sentences),
        len(augmented),
    )
    return augmented


def augment_dictionary_with_phrases(
    entries: list[DictionaryEntry],
    bridge_name: str = "Spanish",
) -> list[DictionaryEntry]:
    """Expand dictionary entries into additional phrase-level examples.

    For entries with example sentences, creates additional entries.
    For entries without, creates simple carrier phrases.
    """
    augmented: list[DictionaryEntry] = list(entries)
    new_count = 0

    for entry in entries:
        if entry.example_source and entry.example_target:
            # Create a parallel pair from the example
            augmented.append(
                DictionaryEntry(
                    headword=entry.example_source,
                    translation=entry.example_target,
                    part_of_speech="phrase",
                    notes=f"Example usage of '{entry.headword}'",
                )
            )
            new_count += 1

    logger.info("Added %d phrase-level entries from examples", new_count)
    return augmented


def create_curriculum_subsets(
    sentences: list[ParallelSentence],
    easy_max_words: int = 8,
    medium_max_words: int = 20,
) -> dict[str, list[ParallelSentence]]:
    """Split parallel sentences into difficulty tiers for curriculum learning.

    Shorter, simpler sentences are learned first, gradually increasing
    complexity. This improves convergence for low-resource scenarios.

    Args:
        sentences: All parallel sentences.
        easy_max_words: Maximum source words for "easy" tier.
        medium_max_words: Maximum source words for "medium" tier.

    Returns:
        Dict with keys 'easy', 'medium', 'hard' containing sentence lists.
    """
    easy: list[ParallelSentence] = []
    medium: list[ParallelSentence] = []
    hard: list[ParallelSentence] = []

    for pair in sentences:
        word_count = len(pair.source.split())
        if word_count <= easy_max_words:
            easy.append(pair)
        elif word_count <= medium_max_words:
            medium.append(pair)
        else:
            hard.append(pair)

    logger.info(
        "Curriculum split: %d easy, %d medium, %d hard",
        len(easy),
        len(medium),
        len(hard),
    )
    return {"easy": easy, "medium": medium, "hard": hard}
