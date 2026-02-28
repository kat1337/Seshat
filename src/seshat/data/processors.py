"""Data cleaning, normalization, and alignment utilities.

Handles the messy reality of low-resource language data: inconsistent
encoding, variant orthographies, misaligned parallel texts, etc.
"""

import logging
import re
import unicodedata
from pathlib import Path

from seshat.data.loaders import DictionaryEntry, ParallelSentence

logger = logging.getLogger(__name__)


def normalize_unicode(text: str) -> str:
    """Normalize Unicode to NFC form (composed characters).

    This ensures consistent representation of accented characters and
    special diacritics common in indigenous language orthographies.
    """
    return unicodedata.normalize("NFC", text)


def clean_text(text: str) -> str:
    """Basic text cleaning: normalize whitespace, strip, normalize unicode."""
    text = normalize_unicode(text)
    text = re.sub(r"\s+", " ", text)  # Collapse multiple whitespace
    text = text.strip()
    return text


def clean_dictionary_entries(
    entries: list[DictionaryEntry],
) -> list[DictionaryEntry]:
    """Clean and validate dictionary entries.

    Removes duplicates, normalizes text, and filters out empty entries.
    """
    seen: set[tuple[str, str]] = set()
    cleaned: list[DictionaryEntry] = []

    for entry in entries:
        hw = clean_text(entry.headword)
        tr = clean_text(entry.translation)

        if not hw or not tr:
            continue

        key = (hw.lower(), tr.lower())
        if key in seen:
            continue
        seen.add(key)

        cleaned.append(
            DictionaryEntry(
                headword=hw,
                translation=tr,
                part_of_speech=clean_text(entry.part_of_speech),
                example_source=clean_text(entry.example_source),
                example_target=clean_text(entry.example_target),
                notes=clean_text(entry.notes),
                variants=entry.variants,
            )
        )

    removed = len(entries) - len(cleaned)
    if removed > 0:
        logger.info("Cleaned dictionary: removed %d duplicates/empty entries", removed)
    return cleaned


def clean_parallel_sentences(
    sentences: list[ParallelSentence],
    min_source_words: int = 2,
    max_source_words: int = 200,
    min_length_ratio: float = 0.2,
    max_length_ratio: float = 5.0,
) -> list[ParallelSentence]:
    """Clean and filter parallel sentence pairs.

    Applies heuristic filters to remove likely misaligned or problematic pairs.

    Args:
        sentences: Input parallel sentences.
        min_source_words: Minimum words in source sentence.
        max_source_words: Maximum words in source sentence.
        min_length_ratio: Minimum ratio of source/target lengths.
        max_length_ratio: Maximum ratio of source/target lengths.

    Returns:
        Filtered list of parallel sentences.
    """
    seen: set[tuple[str, str]] = set()
    cleaned: list[ParallelSentence] = []

    for pair in sentences:
        src = clean_text(pair.source)
        tgt = clean_text(pair.target)

        if not src or not tgt:
            continue

        # Deduplicate
        key = (src, tgt)
        if key in seen:
            continue
        seen.add(key)

        # Length filters
        src_words = len(src.split())
        tgt_words = len(tgt.split())

        if src_words < min_source_words or src_words > max_source_words:
            continue

        if tgt_words == 0:
            continue

        # Length ratio filter (catches misalignment)
        ratio = src_words / tgt_words
        if ratio < min_length_ratio or ratio > max_length_ratio:
            logger.debug(
                "Filtered by length ratio (%.2f): '%s' / '%s'",
                ratio,
                src[:50],
                tgt[:50],
            )
            continue

        cleaned.append(
            ParallelSentence(
                source=src,
                target=tgt,
                domain=pair.domain,
                source_file=pair.source_file,
                confidence=pair.confidence,
            )
        )

    removed = len(sentences) - len(cleaned)
    if removed > 0:
        logger.info("Cleaned parallel data: removed %d entries", removed)
    return cleaned


def segment_sentences(text: str, language: str = "generic") -> list[str]:
    """Split a text block into individual sentences.

    Uses simple heuristics suitable for languages that use period-based
    sentence boundaries. Override for languages with different conventions.

    Args:
        text: Input text block.
        language: Language hint for future language-specific segmentation.

    Returns:
        List of individual sentences.
    """
    # Split on common sentence-ending punctuation followed by space or newline
    # This is a simple heuristic — may need language-specific rules
    sentences = re.split(r"(?<=[.!?。])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def align_bitext(
    source_text: str,
    target_text: str,
    method: str = "sentence",
) -> list[ParallelSentence]:
    """Attempt to align two text blocks into parallel sentences.

    Simple sentence-count-based alignment. For better results, use
    external alignment tools (e.g., hunalign, bleualign).

    Args:
        source_text: Text in target language.
        target_text: Text in bridge language.
        method: Alignment method ('sentence' for simple 1:1).

    Returns:
        List of aligned ParallelSentence pairs.
    """
    src_sentences = segment_sentences(source_text)
    tgt_sentences = segment_sentences(target_text)

    pairs: list[ParallelSentence] = []

    if method == "sentence":
        # Simple 1:1 alignment — only works if texts are already sentence-aligned
        min_len = min(len(src_sentences), len(tgt_sentences))
        if abs(len(src_sentences) - len(tgt_sentences)) > min_len * 0.1:
            logger.warning(
                "Sentence count mismatch (%d vs %d) — alignment may be poor",
                len(src_sentences),
                len(tgt_sentences),
            )

        for src, tgt in zip(src_sentences[:min_len], tgt_sentences[:min_len]):
            pairs.append(
                ParallelSentence(
                    source=src,
                    target=tgt,
                    confidence=0.7,  # Lower confidence for auto-alignment
                )
            )

    return pairs


def compute_data_statistics(
    dictionary: list[DictionaryEntry],
    parallel: list[ParallelSentence],
    monolingual: list,
) -> dict:
    """Compute statistics about the loaded dataset for reporting."""
    stats: dict = {
        "dictionary_entries": len(dictionary),
        "parallel_sentences": len(parallel),
        "monolingual_segments": len(monolingual),
    }

    if parallel:
        src_lengths = [len(p.source.split()) for p in parallel]
        tgt_lengths = [len(p.target.split()) for p in parallel]
        stats["parallel_avg_source_length"] = sum(src_lengths) / len(src_lengths)
        stats["parallel_avg_target_length"] = sum(tgt_lengths) / len(tgt_lengths)
        stats["parallel_max_source_length"] = max(src_lengths)

    if dictionary:
        unique_headwords = len({e.headword.lower() for e in dictionary})
        stats["unique_headwords"] = unique_headwords
        has_pos = sum(1 for e in dictionary if e.part_of_speech)
        stats["entries_with_pos"] = has_pos

    return stats
