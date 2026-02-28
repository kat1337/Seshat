# Data Requirements for Low-Resource Language Fine-Tuning

## Executive Summary

Teaching an LLM a new language via LoRA/QLoRA fine-tuning is fundamentally different from training from scratch. We're leveraging the model's existing multilingual and linguistic capabilities — it already understands grammar, translation, morphology, and language structure. We just need to teach it a new mapping.

**Bottom line: With your current ~1,000 pages of transcriptions/translations plus a dictionary, you have enough data to build a functional prototype.** Full fluency requires more, but you can get useful translation and basic linguistic analysis working.

---

## What You Currently Have (Estimated)

| Resource | Estimated Volume | Estimated Yield |
|----------|-----------------|-----------------|
| Dictionary | Unknown size | 2,000–15,000 structured entries |
| Transcriptions + translations (~1,000 pages) | ~300,000 words | 30,000–60,000 sentence pairs |
| Unprocessed scans | Unknown | TBD after OCR |
| Unprocessed audio | Unknown | TBD after transcription |

### Estimations Methodology
- Average page: ~300 words, ~25-35 sentences (bilingual = ~15 sentence pairs per page)
- Dictionary entries vary enormously. A typical field dictionary has 2,000-15,000 headwords
- Scans and audio are wildcards — could double your corpus or yield little usable data

---

## Data Tiers: What's Needed for What

### Tier 1: Basic Translation (Minimum Viable)
**Goal**: Translate simple sentences between the language and Spanish with reasonable accuracy.

| Data Type | Amount | Purpose |
|-----------|--------|---------|
| Parallel sentence pairs | 10,000–20,000 | Core translation training |
| Dictionary entries (structured) | 2,000+ | Vocabulary coverage |
| Monolingual sentences (L1) | 5,000 | Language modeling / fluency |

**Expected quality**: Handles simple, common sentence patterns. Struggles with complex grammar, rare vocabulary, and idiomatic expressions. BLEU scores in the 15-25 range.

**Your status**: ✅ You likely have enough or close to enough for this tier.

### Tier 2: Reliable Translation + Basic Linguistic Analysis
**Goal**: Accurate translation of most sentence types. Can explain basic grammar and word meanings.

| Data Type | Amount | Purpose |
|-----------|--------|---------|
| Parallel sentence pairs | 30,000–60,000 | Broader translation coverage |
| Dictionary entries | 5,000–10,000 | Better vocabulary coverage |
| Monolingual sentences | 20,000+ | Fluency and generation |
| Grammar explanation examples | 500–1,000 | Teach the model to explain grammar |
| Morphological breakdowns | 200–500 | Morphology analysis capability |

**Expected quality**: Solid translations for most text types. Can explain word meanings and basic grammatical structures. BLEU scores 25-35.

**Your status**: ⚠️ You may be in this range with existing data. Processing scans and audio would solidify this.

### Tier 3: Fluent Translation + Linguistic Research Tool
**Goal**: Near-fluent translation. Detailed grammatical analysis. Useful for academic linguistics research.

| Data Type | Amount | Purpose |
|-----------|--------|---------|
| Parallel sentence pairs | 100,000–200,000 | Comprehensive translation |
| Dictionary entries | 10,000–30,000 | Full vocabulary |
| Monolingual text | 50,000+ sentences | Natural fluency |
| Grammar explanations | 2,000–5,000 | Detailed grammatical instruction |
| Morphological analyses | 1,000+ | Systematic morphology |
| Linguistic Q&A pairs | 1,000–5,000 | Research-oriented responses |
| Contextual/discourse examples | 1,000+ | Pragmatics and register |

**Expected quality**: Fluent translations of most text. Can produce detailed grammatical explanations, morphological analysis, and engage in linguistic discussion. BLEU 35+.

**Your status**: ❌ Requires significant additional data, likely from OCR/audio processing + synthetic data augmentation.

---

## Data Type Breakdown

### 1. Parallel Sentence Pairs (Most Important)

These are aligned sentence-by-sentence translations: one sentence in the target language, one in Spanish.

**Sources from your data**:
- Translated texts (direct extraction)
- Dictionary example sentences
- Transcriptions with translations

**Format after processing**:
```json
{
  "source": "indigenous text here",
  "target": "spanish translation here",
  "metadata": {
    "origin": "transcription_book_3",
    "page": 45,
    "confidence": "high",
    "domain": "narrative"
  }
}
```

**Quality requirements**:
- Sentence-level alignment (not paragraph-level)
- Consistent orthography in the indigenous language
- Clean Spanish (no OCR artifacts, abbreviations expanded)
- Domain diversity (narrative, conversational, descriptive, procedural)

### 2. Dictionary Entries (Structured)

Transform dictionary into structured training data with multiple training signals per entry.

**One dictionary entry generates multiple training examples**:
- Word → definition (lookup)
- Definition → word (reverse lookup)
- Word → example sentence (usage)
- Word → part of speech + morphological class
- Word → related words / etymological notes

**A 5,000-entry dictionary can yield 15,000-25,000 training examples.**

### 3. Monolingual Text (Target Language Only)

Even text without translations is valuable — it teaches the model the "shape" of the language: word order, common patterns, phonotactics (in written form).

**Sources**: Any indigenous-language text, even without translation. Portions of transcriptions where only L1 is available.

### 4. Grammar Explanation Examples (Synthetic + Manual)

Since no formal grammar exists, you need to CREATE training examples that teach grammatical concepts. This is where **synthetic data generation** is critical.

**Strategy**:
1. Identify grammatical patterns from the parallel corpus (common suffixes, word order patterns, reduplication, etc.)
2. Use Claude or GPT-4 to generate explanations for these patterns based on the examples
3. Have a linguist review and correct the synthetic explanations
4. Feed corrected explanations back as training data

**Example**:
```json
{
  "messages": [
    {"role": "user", "content": "What grammatical role does the suffix '-kuna' play?"},
    {"role": "assistant", "content": "The suffix '-kuna' is a pluralizer in {LANG}. It attaches to nouns to indicate plurality. Examples: 'wasi' (house) → 'wasikuna' (houses), 'runa' (person) → 'runakuna' (people). It appears to be productive and applies to both animate and inanimate nouns."}
  ]
}
```

### 5. Linguistic Analysis Q&A Pairs

For the research use case, you need examples of the model answering the kinds of questions linguists ask:
- "What is the basic word order in {LANG}?"
- "How does {LANG} express negation?"
- "What are the evidentiality markers in {LANG}?"
- "Compare the morphological structure of these two sentences"

These can be initially generated synthetically and refined with expert review.

---

## Data Augmentation Strategies

Given limited data, augmentation is essential:

### 1. Back-Translation
- Translate Spanish text → indigenous language using your early model
- Filter for quality, add good translations to training set
- Iterate: each training round produces better back-translations

### 2. Dictionary Expansion
- Generate example sentences for dictionary entries using the model
- Create fill-in-the-blank exercises from existing sentences
- Generate synonyms and paraphrases

### 3. Synthetic Grammar Examples
- Use a strong model (Claude, GPT-4) to analyze patterns in your parallel corpus
- Generate grammatical explanations and morphological breakdowns
- Expert review is critical — AI will make mistakes about the target language

### 4. Sentence Permutation
- Shuffle word order where grammatically acceptable to create variants
- Substitute vocabulary items in template sentences
- Create minimal pairs for contrastive grammar learning

### 5. Cross-Lingual Transfer
- If related languages have more data, use them for pre-training
- Amazonic language families (Tupian, Arawakan, Cariban, etc.) share features that can bootstrap learning

---

## Data Quality Checklist

Before training, verify:

- [ ] **Orthographic consistency**: Is the indigenous language spelled consistently throughout? Multiple transcription systems are common for indigenous languages — normalize to one
- [ ] **Alignment quality**: Are sentence pairs actually translations of each other? Check a random sample of 100
- [ ] **No data leakage**: Test set must not overlap with training set at the sentence level
- [ ] **Domain balance**: Not 100% narratives or 100% conversational. Mix domains
- [ ] **Tokenizer compatibility**: Check that the base model's tokenizer handles the language reasonably (token fertility analysis)
- [ ] **Encoding**: All text in UTF-8. Indigenous languages often use special characters — verify they survive the pipeline
- [ ] **Deduplication**: Remove duplicate sentence pairs
- [ ] **Length distribution**: Check for extreme outliers (very short or very long sentences)

---

## Comparison: Speaker-Aware Translation (Manga JP→EN)

For reference on the manga speaker-aware translation project:

Speaker-aware translation adds an additional challenge layer — the model must track who is speaking and adapt voice/register accordingly. This requires:

| Data Type | Estimated Minimum |
|-----------|-------------------|
| Dialogue-tagged parallel sentences | 50,000+ |
| Speaker profile descriptions | 500+ characters |
| Style-differentiated translations of same content | 5,000+ pairs |
| Context windows (surrounding dialogue) | Included with each example |

The key difference is that speaker-aware translation needs **metadata-rich** examples (who speaks, to whom, emotional tone, formality level) rather than just raw volume. The indigenous language project should adopt similar metadata schemas from the start, even if speaker awareness isn't the immediate priority.

---

## Timeline Estimation

| Phase | Duration | Data Required |
|-------|----------|--------------|
| Data audit and cleaning | 1-2 weeks | Existing data |
| Tokenizer analysis | 1-2 days | Sample of target language text |
| First training run (Tier 1) | 2-3 days compute | 10k+ sentence pairs |
| OCR processing (if pursued) | 1-2 weeks | Scanned documents |
| Audio transcription (if pursued) | 1-3 weeks | Audio recordings |
| Second training run (Tier 2) | 3-5 days compute | 30k+ sentence pairs |
| Synthetic data generation + review | 2-4 weeks ongoing | Expert reviewer time |
| Tier 3 target | 2-6 months total | Full pipeline + augmentation |
