"""Tests for data loading and processing."""

import tempfile
from pathlib import Path

from seshat.data.loaders import (
    DictionaryEntry,
    ParallelSentence,
    load_dictionary_csv,
    load_parallel_texts,
)
from seshat.data.processors import (
    clean_dictionary_entries,
    clean_parallel_sentences,
    clean_text,
    normalize_unicode,
)


def test_normalize_unicode():
    # NFC normalization: combining sequences → precomposed
    text = "café"  # Could be e + combining acute
    result = normalize_unicode(text)
    assert isinstance(result, str)
    assert len(result) > 0


def test_clean_text():
    assert clean_text("  hello   world  ") == "hello world"
    assert clean_text("") == ""
    assert clean_text("  ") == ""


def test_clean_dictionary_deduplication():
    entries = [
        DictionaryEntry(headword="yaku", translation="agua"),
        DictionaryEntry(headword="Yaku", translation="Agua"),  # Duplicate (case-insensitive)
        DictionaryEntry(headword="killa", translation="luna"),
    ]
    cleaned = clean_dictionary_entries(entries)
    assert len(cleaned) == 2


def test_clean_dictionary_empty_removal():
    entries = [
        DictionaryEntry(headword="yaku", translation="agua"),
        DictionaryEntry(headword="", translation="agua"),  # Empty headword
        DictionaryEntry(headword="killa", translation=""),  # Empty translation
    ]
    cleaned = clean_dictionary_entries(entries)
    assert len(cleaned) == 1


def test_clean_parallel_sentences():
    sentences = [
        ParallelSentence(source="yaku hatun", target="el agua es grande"),
        ParallelSentence(source="a", target="very long sentence that is disproportionate"),  # Too short
        ParallelSentence(source="yaku hatun", target="el agua es grande"),  # Duplicate
    ]
    cleaned = clean_parallel_sentences(sentences, min_source_words=2)
    assert len(cleaned) == 1


def test_load_dictionary_csv():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
        f.write("headword,translation,part_of_speech\n")
        f.write("yaku,agua,noun\n")
        f.write("killa,luna,noun\n")
        f.write("hatun,grande,adjective\n")
        tmp_path = f.name

    entries = load_dictionary_csv(
        Path(tmp_path),
        source_col="headword",
        target_col="translation",
        pos_col="part_of_speech",
    )
    assert len(entries) == 3
    assert entries[0].headword == "yaku"
    assert entries[0].translation == "agua"
    assert entries[0].part_of_speech == "noun"


def test_load_parallel_texts():
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "text1.src"
        tgt_path = Path(tmpdir) / "text1.tgt"

        src_path.write_text("yaku hatun\nkilla sumaq\n", encoding="utf-8")
        tgt_path.write_text("el agua es grande\nla luna es hermosa\n", encoding="utf-8")

        pairs = load_parallel_texts(Path(tmpdir), source_suffix=".src", target_suffix=".tgt")
        assert len(pairs) == 2
        assert pairs[0].source == "yaku hatun"
        assert pairs[0].target == "el agua es grande"
