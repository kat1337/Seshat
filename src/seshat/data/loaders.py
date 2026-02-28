"""Load raw linguistic data from various source formats.

Supports dictionaries (CSV, JSON), parallel texts, monolingual transcriptions,
and produces standardized internal representations for downstream processing.
"""

import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


@dataclass
class DictionaryEntry:
    """A single dictionary entry with headword, translation, and metadata."""

    headword: str
    translation: str
    part_of_speech: str = ""
    example_source: str = ""
    example_target: str = ""
    notes: str = ""
    variants: list[str] = field(default_factory=list)


@dataclass
class ParallelSentence:
    """A sentence-aligned translation pair."""

    source: str  # Target language
    target: str  # Bridge language (e.g., Spanish)
    domain: str = ""
    source_file: str = ""
    confidence: float = 1.0  # Manual alignment = 1.0, auto-aligned = lower


@dataclass
class MonolingualText:
    """A monolingual text passage in the target language."""

    text: str
    source_file: str = ""
    domain: str = ""


@dataclass
class LanguageConfig:
    """Configuration for a specific language, loaded from YAML."""

    code: str
    name: str
    bridge_language: str
    bridge_language_name: str
    family: str = ""
    region: str = ""
    prompts: dict[str, str] = field(default_factory=dict)
    data_config: dict = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Path) -> "LanguageConfig":
        """Load language configuration from a YAML file."""
        with open(path) as f:
            config = yaml.safe_load(f)

        lang = config["language"]
        prompts = config.get("prompts", {})

        # Format prompt templates with language info
        formatted_prompts = {}
        for key, template in prompts.items():
            formatted_prompts[key] = template.format(**lang)

        return cls(
            code=lang["code"],
            name=lang["name"],
            bridge_language=lang["bridge_language"],
            bridge_language_name=lang["bridge_language_name"],
            family=lang.get("family", ""),
            region=lang.get("region", ""),
            prompts=formatted_prompts,
            data_config=config.get("data", {}),
        )


def load_dictionary_csv(
    path: Path,
    source_col: str = "headword",
    target_col: str = "translation",
    pos_col: str | None = "part_of_speech",
    example_col: str | None = "example",
    notes_col: str | None = "notes",
    encoding: str = "utf-8",
    separator: str = ",",
) -> list[DictionaryEntry]:
    """Load dictionary entries from a CSV file.

    Args:
        path: Path to CSV file.
        source_col: Column name for headword in target language.
        target_col: Column name for translation in bridge language.
        pos_col: Column name for part of speech (optional).
        example_col: Column name for example sentence (optional).
        notes_col: Column name for notes (optional).
        encoding: File encoding.
        separator: CSV separator character.

    Returns:
        List of DictionaryEntry objects.
    """
    entries: list[DictionaryEntry] = []
    path = Path(path)

    with open(path, encoding=encoding, newline="") as f:
        reader = csv.DictReader(f, delimiter=separator)
        for i, row in enumerate(reader):
            headword = row.get(source_col, "").strip()
            translation = row.get(target_col, "").strip()

            if not headword or not translation:
                logger.warning("Skipping empty entry at row %d in %s", i + 1, path)
                continue

            entry = DictionaryEntry(
                headword=headword,
                translation=translation,
                part_of_speech=row.get(pos_col, "").strip() if pos_col else "",
                example_source=row.get(example_col, "").strip() if example_col else "",
                notes=row.get(notes_col, "").strip() if notes_col else "",
            )
            entries.append(entry)

    logger.info("Loaded %d dictionary entries from %s", len(entries), path)
    return entries


def load_dictionary_json(path: Path) -> list[DictionaryEntry]:
    """Load dictionary entries from a JSON file.

    Expected format: list of objects with at minimum 'headword' and 'translation' keys.
    """
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    entries = []
    for item in data:
        if isinstance(item, dict) and "headword" in item and "translation" in item:
            entries.append(
                DictionaryEntry(
                    headword=item["headword"],
                    translation=item["translation"],
                    part_of_speech=item.get("part_of_speech", ""),
                    example_source=item.get("example_source", ""),
                    example_target=item.get("example_target", ""),
                    notes=item.get("notes", ""),
                    variants=item.get("variants", []),
                )
            )

    logger.info("Loaded %d dictionary entries from %s", len(entries), path)
    return entries


def load_parallel_texts(
    directory: Path,
    source_suffix: str = ".src",
    target_suffix: str = ".tgt",
    encoding: str = "utf-8",
) -> list[ParallelSentence]:
    """Load sentence-aligned parallel texts from paired files.

    Expects pairs of files with matching names but different suffixes,
    where line N in the source file aligns with line N in the target file.

    Args:
        directory: Directory containing paired text files.
        source_suffix: File suffix for target language files.
        target_suffix: File suffix for bridge language files.
        encoding: File encoding.

    Returns:
        List of ParallelSentence objects.
    """
    directory = Path(directory)
    pairs: list[ParallelSentence] = []

    source_files = sorted(directory.glob(f"*{source_suffix}"))

    for src_file in source_files:
        tgt_file = src_file.with_suffix(target_suffix)
        if not tgt_file.exists():
            logger.warning("No matching target file for %s", src_file)
            continue

        with open(src_file, encoding=encoding) as sf, open(tgt_file, encoding=encoding) as tf:
            source_lines = sf.readlines()
            target_lines = tf.readlines()

        if len(source_lines) != len(target_lines):
            logger.warning(
                "Line count mismatch: %s (%d) vs %s (%d)",
                src_file.name,
                len(source_lines),
                tgt_file.name,
                len(target_lines),
            )
            # Use minimum length
            min_len = min(len(source_lines), len(target_lines))
            source_lines = source_lines[:min_len]
            target_lines = target_lines[:min_len]

        for src_line, tgt_line in zip(source_lines, target_lines):
            src_text = src_line.strip()
            tgt_text = tgt_line.strip()
            if src_text and tgt_text:
                pairs.append(
                    ParallelSentence(
                        source=src_text,
                        target=tgt_text,
                        source_file=src_file.name,
                    )
                )

    logger.info("Loaded %d parallel sentence pairs from %s", len(pairs), directory)
    return pairs


def load_monolingual_texts(
    directory: Path,
    glob_pattern: str = "*.txt",
    encoding: str = "utf-8",
) -> list[MonolingualText]:
    """Load monolingual texts from a directory of text files.

    Each non-empty line becomes a separate MonolingualText entry.
    """
    directory = Path(directory)
    texts: list[MonolingualText] = []

    for txt_file in sorted(directory.glob(glob_pattern)):
        with open(txt_file, encoding=encoding) as f:
            for line in f:
                text = line.strip()
                if text:
                    texts.append(
                        MonolingualText(text=text, source_file=txt_file.name)
                    )

    logger.info("Loaded %d text segments from %s", len(texts), directory)
    return texts


def load_all_data(
    lang_config: LanguageConfig,
    data_root: Path,
) -> dict:
    """Load all available data for a language based on its configuration.

    Args:
        lang_config: Language configuration object.
        data_root: Root data directory (e.g., data/raw/).

    Returns:
        Dict with keys 'dictionary', 'parallel', 'monolingual', each containing
        lists of the respective dataclass objects.
    """
    data_root = Path(data_root)
    result: dict = {"dictionary": [], "parallel": [], "monolingual": []}
    dc = lang_config.data_config

    # Load dictionary
    if "dictionary" in dc:
        dict_path = data_root / dc["dictionary"]["path"]
        if dict_path.exists():
            fmt = dc["dictionary"].get("format", "csv")
            if fmt == "csv":
                cols = dc["dictionary"].get("columns", {})
                result["dictionary"] = load_dictionary_csv(
                    dict_path,
                    source_col=cols.get("source", "headword"),
                    target_col=cols.get("target", "translation"),
                    pos_col=cols.get("pos"),
                    example_col=cols.get("example"),
                    notes_col=cols.get("notes"),
                    encoding=dc["dictionary"].get("encoding", "utf-8"),
                    separator=dc["dictionary"].get("separator", ","),
                )
            elif fmt == "json":
                for json_file in dict_path.glob("*.json"):
                    result["dictionary"].extend(load_dictionary_json(json_file))

    # Load parallel texts
    if "parallel_texts" in dc:
        par_path = data_root / dc["parallel_texts"]["path"]
        if par_path.exists():
            result["parallel"] = load_parallel_texts(
                par_path,
                source_suffix=dc["parallel_texts"].get("source_suffix", ".src"),
                target_suffix=dc["parallel_texts"].get("target_suffix", ".tgt"),
            )

    # Load monolingual transcriptions
    if "transcriptions" in dc:
        mono_path = data_root / dc["transcriptions"]["path"]
        if mono_path.exists():
            result["monolingual"] = load_monolingual_texts(mono_path)

    logger.info(
        "Total data loaded — dictionary: %d, parallel: %d, monolingual: %d",
        len(result["dictionary"]),
        len(result["parallel"]),
        len(result["monolingual"]),
    )
    return result
