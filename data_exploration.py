"""
data_exploration.py
EDA for Low-Resource Language Corpus (.json format)
Supports: cham.json, khmer.json, tay_nung.json (or any similar JSON corpus)
"""

import json
import os
import re
import argparse
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from wordcloud import WordCloud
from tqdm import tqdm
from transformers import AutoTokenizer

warnings.filterwarnings("ignore")

# ── Aesthetic config ──────────────────────────────────────────────────────────
PALETTE   = sns.color_palette("tab10")
BG_COLOR  = "#F8F9FA"
GRID_COLOR= "#E0E0E0"
plt.rcParams.update({
    "figure.facecolor":  BG_COLOR,
    "axes.facecolor":    BG_COLOR,
    "axes.grid":         True,
    "grid.color":        GRID_COLOR,
    "grid.linestyle":    "--",
    "grid.linewidth":    0.6,
    "font.family":       "DejaVu Sans",
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
})

OUTPUT_DIR = Path("eda_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# ═════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ═════════════════════════════════════════════════════════════════════════════

def load_json_corpus(filepath: str) -> list[dict]:
    """Load a JSON corpus file (list of records or single record)."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]
    print(f"  Loaded {len(data):,} records from {filepath}")
    return data


def load_multiple_corpora(data_dir: str, lang_files: dict[str, str] = None) -> dict[str, list[dict]]:
    """
    Load several language corpora from a directory.
    lang_files: {language_label: filename}, e.g. {"Cham": "cham.json"}
    If None, auto-detect all .json files in data_dir.
    """
    corpora = {}
    data_path = Path(data_dir)

    if lang_files:
        for lang, fname in lang_files.items():
            fp = data_path / fname
            if fp.exists():
                corpora[lang] = load_json_corpus(str(fp))
            else:
                print(f"  [WARN] {fp} not found – skipping.")
    else:
        for fp in sorted(data_path.glob("*.json")):
            lang = fp.stem.replace("_", " ").title()
            corpora[lang] = load_json_corpus(str(fp))

    return corpora


# ═════════════════════════════════════════════════════════════════════════════
# 2. LANGUAGE DETECTION & SENTENCE SPLITTING
# ═════════════════════════════════════════════════════════════════════════════

_KHMER_RE   = re.compile(r'[\u1780-\u17FF]')
_SENT_KHMER = re.compile(r'[។៕]+')
_SENT_LATIN = re.compile(r'(?<=[^\d])[.!?…]+(?=\s|$)|(?<=\n)\n+')


def detect_script(text: str) -> str:
    return "khmer" if _KHMER_RE.search(text) else "latin"


def split_sentences(text: str, script: str) -> list[str]:
    if not text:
        return []
    if script == "khmer":
        parts = _SENT_KHMER.split(text)
    else:
        normalised = re.sub(r'\n{2,}', '. ', text)
        parts = _SENT_LATIN.split(normalised)
    return [p.strip() for p in parts if p.strip()]


# ═════════════════════════════════════════════════════════════════════════════
# 2-B. WORD SEGMENTATION
# ═════════════════════════════════════════════════════════════════════════════

_khmer_word_seg = None


def _load_khmer_segmenter():
    global _khmer_word_seg
    if _khmer_word_seg is not None:
        return _khmer_word_seg
    try:
        from khmernltk import word_tokenize as kh_tokenize
        _khmer_word_seg = kh_tokenize
        print("  [Khmer] khmernltk word segmenter loaded ✓")
    except ImportError:
        print(
            "  [WARN] khmernltk not installed — Khmer will fall back to "
            "character-cluster tokenisation (counts will be inflated).\n"
            "  Install with:  pip install khmernltk"
        )
        _khmer_word_seg = _khmer_cluster_fallback
    return _khmer_word_seg


def _khmer_cluster_fallback(text: str) -> list[str]:
    tokens = re.findall(r'[\u1780-\u17FF]+', text)
    return [t for t in tokens if t.strip()]


def tokenize(text: str, script: str = "latin") -> list[str]:
    if not text:
        return []
    if script == "khmer":
        seg = _load_khmer_segmenter()
        raw_tokens = seg(text)
        return [
            t.strip().lower()
            for t in raw_tokens
            if t.strip() and not re.fullmatch(r'[\s\u200b\u00a0]+', t)
               and re.search(r'[\u1780-\u17FF]', t)
        ]
    else:
        return re.findall(r'\b\w+\b', text.lower(), flags=re.UNICODE)


# ═════════════════════════════════════════════════════════════════════════════
# 3. XLM-RoBERTa BPE TOKENIZER
# ═════════════════════════════════════════════════════════════════════════════

_XLM_MODEL_NAME = "xlm-roberta-base"
_bpe_tokenizer  = None


def get_bpe_tokenizer() -> AutoTokenizer:
    global _bpe_tokenizer
    if _bpe_tokenizer is None:
        print(f"  Loading XLM-RoBERTa tokenizer ({_XLM_MODEL_NAME}) …")
        _bpe_tokenizer = AutoTokenizer.from_pretrained(
            _XLM_MODEL_NAME,
            use_fast=True,
        )
        print(f"  Tokenizer ready  |  vocab size: {_bpe_tokenizer.vocab_size:,}")
    return _bpe_tokenizer


def bpe_tokenize_corpus(
    records:  list[dict],
    lang:     str,
    batch_sz: int = 64,
) -> list[dict]:
    """
    Run XLM-RoBERTa BPE tokenizer over every record's `content` field.
    Also computes per-word fragmentation info for vocab extension metrics.
    """
    tok = get_bpe_tokenizer()
    results = []

    texts = [rec.get("content", "") or "" for rec in records]

    print(f"\n  [{lang}] BPE-tokenising {len(texts):,} documents "
          f"(batch={batch_sz}, model={_XLM_MODEL_NAME}) …")

    for i in tqdm(
        range(0, len(texts), batch_sz),
        desc=f"  BPE [{lang}]",
        unit="batch",
        dynamic_ncols=True,
        colour="cyan",
    ):
        batch = texts[i : i + batch_sz]
        encoded = tok(
            batch,
            add_special_tokens=False,
            truncation=False,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        for ids in encoded["input_ids"]:
            subwords = tok.convert_ids_to_tokens(ids)
            results.append({
                "bpe_token_count":   len(ids),
                "bpe_unique_tokens": len(set(ids)),
                "bpe_tokens":        subwords,
            })

    print(f"  [{lang}] ✓ BPE tokenisation done — "
          f"{sum(r['bpe_token_count'] for r in results):,} total subword tokens")
    return results


# ═════════════════════════════════════════════════════════════════════════════
# 3-B. VOCABULARY EXTENSION METRICS
# ═════════════════════════════════════════════════════════════════════════════

def compute_vocab_extension_metrics(
    words: list[str],
    bpe_tokens: list[str],
) -> dict:
    """
    Compute 3 key metrics to assess whether a low-resource language
    needs vocabulary extension in XLM-RoBERTa:

    1. Fragmentation Ratio  = total BPE tokens / total whitespace words
       (> 2.0 indicates poor tokenizer coverage → extend vocab)

    2. Long-Split Ratio     = fraction of words split into > 3 subword pieces
       (> 0.3 = 30% of words need >3 pieces → high OOV signal)

    3. Tokenizer Coverage   = fraction of words tokenized as a single token
       (< 0.5 = less than 50% single-token words → low coverage)

    Returns a dict with all three metrics plus intermediate counts.
    """
    if not words or not bpe_tokens:
        return {
            "fragmentation_ratio": float("nan"),
            "long_split_ratio":    float("nan"),
            "tokenizer_coverage":  float("nan"),
            "n_words":             0,
            "n_bpe_tokens":        0,
            "n_single_token_words":0,
            "n_long_split_words":  0,
        }

    n_words     = len(words)
    n_bpe       = len(bpe_tokens)
    frag_ratio  = n_bpe / max(n_words, 1)

    # Reconstruct per-word split counts by rejoining subwords
    # XLM-RoBERTa uses "▁" prefix to mark word starts (SentencePiece)
    per_word_pieces = []
    current_count   = 0
    for token in bpe_tokens:
        if token.startswith("▁") or current_count == 0:
            if current_count > 0:
                per_word_pieces.append(current_count)
            current_count = 1
        else:
            current_count += 1
    if current_count > 0:
        per_word_pieces.append(current_count)

    if not per_word_pieces:
        per_word_pieces = [1] * n_words   # fallback

    n_reconstructed    = len(per_word_pieces)
    n_single           = sum(1 for c in per_word_pieces if c == 1)
    n_long_split       = sum(1 for c in per_word_pieces if c > 3)

    coverage           = n_single    / max(n_reconstructed, 1)
    long_split_ratio   = n_long_split / max(n_reconstructed, 1)

    return {
        "fragmentation_ratio":  frag_ratio,
        "long_split_ratio":     long_split_ratio,
        "tokenizer_coverage":   coverage,
        "n_words":              n_words,
        "n_bpe_tokens":         n_bpe,
        "n_single_token_words": n_single,
        "n_long_split_words":   n_long_split,
    }


# ═════════════════════════════════════════════════════════════════════════════
# 4. FEATURE EXTRACTION
# ═════════════════════════════════════════════════════════════════════════════

def extract_features(records: list[dict], bpe_results: list[dict] | None = None) -> pd.DataFrame:
    """
    Extract text-level features from corpus records.
    If bpe_results is provided (same length as records), BPE columns are merged in.
    """
    rows = []
    seg_desc = "  Segment [Khmer]" if any(
        detect_script(r.get("content","") or "") == "khmer" for r in records[:1]
    ) else "  Tokenise [Latin]"
    for idx, rec in enumerate(tqdm(
        records,
        desc=seg_desc,
        unit="doc",
        dynamic_ncols=True,
        colour="green",
    )):
        content = rec.get("content", "") or ""
        title   = rec.get("title",   "") or ""
        summary = rec.get("summary", "") or ""
        tags    = rec.get("tags",    []) or []
        cat     = rec.get("category","Unknown")
        site    = rec.get("site",    "Unknown")

        full_text = content

        script     = detect_script(full_text)
        words      = tokenize(full_text, script=script)
        sentences  = split_sentences(full_text, script)
        paragraphs = [p.strip() for p in full_text.split('\n') if p.strip()]

        sent_word_counts = [len(tokenize(s, script=script)) for s in sentences]

        row = {
            "site":             site,
            "category":         cat,
            "n_tags":           len(tags),
            "title_len":        len(title.split()),
            "script":           script,
            "char_count":       len(full_text),
            "word_count":       len(words),
            "unique_words":     len(set(words)),
            "sent_count":       len(sentences),
            "para_count":       len(paragraphs),
            "type_token_ratio": len(set(words)) / max(len(words), 1),
            "avg_word_len":     np.mean([len(w) for w in words]) if words else 0,
            "avg_sent_len":     np.mean(sent_word_counts) if sent_word_counts else 0,
            "_words":           words,
            "_text":            full_text,
            "_sentences":       sentences,
        }

        # ── Merge BPE columns if available ────────────────────────────────────
        if bpe_results is not None:
            bpe = bpe_results[idx]
            row["bpe_token_count"]          = bpe["bpe_token_count"]
            row["bpe_unique_tokens"]        = bpe["bpe_unique_tokens"]
            row["bpe_fertility"]            = (bpe["bpe_token_count"]
                                               / max(len(words), 1))
            row["bpe_type_token_ratio"]     = (bpe["bpe_unique_tokens"]
                                               / max(bpe["bpe_token_count"], 1))
            row["_bpe_tokens"]              = bpe["bpe_tokens"]

            # ── Vocabulary extension metrics per document ──────────────────
            vm = compute_vocab_extension_metrics(words, bpe["bpe_tokens"])
            row["vocab_fragmentation_ratio"] = vm["fragmentation_ratio"]
            row["vocab_long_split_ratio"]    = vm["long_split_ratio"]
            row["vocab_tokenizer_coverage"]  = vm["tokenizer_coverage"]
        else:
            row["bpe_token_count"]           = np.nan
            row["bpe_unique_tokens"]         = np.nan
            row["bpe_fertility"]             = np.nan
            row["bpe_type_token_ratio"]      = np.nan
            row["_bpe_tokens"]               = []
            row["vocab_fragmentation_ratio"] = np.nan
            row["vocab_long_split_ratio"]    = np.nan
            row["vocab_tokenizer_coverage"]  = np.nan

        rows.append(row)
    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════════
# 4-B. SAVE PER-LANGUAGE JSON STATS
# ═════════════════════════════════════════════════════════════════════════════

def _safe_float(v):
    """Convert numpy/pandas scalar to plain Python float, handle NaN."""
    try:
        f = float(v)
        return None if np.isnan(f) else round(f, 6)
    except Exception:
        return None


def _safe_int(v):
    try:
        return int(v)
    except Exception:
        return None


def build_lang_stats(lang: str, df: pd.DataFrame) -> dict:
    """
    Build a comprehensive JSON-serialisable stats dict for one language.
    Includes corpus stats, BPE stats, vocab extension metrics,
    top words, category distribution, and per-doc percentile data.
    """
    all_words = [w for ws in df["_words"] for w in ws]
    vocab     = set(all_words)
    freq      = Counter(all_words)
    script    = df["script"].mode()[0] if "script" in df.columns else "latin"

    # ── Corpus stats ──────────────────────────────────────────────────────────
    corpus_stats = {
        "n_documents":       _safe_int(len(df)),
        "total_tokens":      _safe_int(len(all_words)),
        "vocabulary_size":   _safe_int(len(vocab)),
        "script":            script,
        "sentence_splitter": "Khmer ។/៕" if script == "khmer" else "Latin .!?…",
    }

    def describe_col(col):
        s = df[col].dropna()
        if len(s) == 0:
            return {}
        return {
            "mean":   _safe_float(s.mean()),
            "median": _safe_float(s.median()),
            "std":    _safe_float(s.std()),
            "min":    _safe_float(s.min()),
            "max":    _safe_float(s.max()),
            "q1":     _safe_float(s.quantile(0.25)),
            "q3":     _safe_float(s.quantile(0.75)),
        }

    text_metrics = {
        col: describe_col(col)
        for col in [
            "word_count", "unique_words", "char_count",
            "sent_count", "para_count", "type_token_ratio",
            "avg_word_len", "avg_sent_len",
        ]
    }

    # ── BPE stats ──────────────────────────────────────────────────────────────
    bpe_stats = {}
    if df["bpe_token_count"].notna().any():
        all_bpe = [t for toks in df["_bpe_tokens"] for t in toks]
        bpe_freq = Counter(all_bpe)
        bpe_stats = {
            "model":                "xlm-roberta-base",
            "special_tokens":       "excluded",
            "total_bpe_tokens":     _safe_int(int(df["bpe_token_count"].sum())),
            "unique_bpe_vocab":     _safe_int(len(bpe_freq)),
            "bpe_token_count":      describe_col("bpe_token_count"),
            "bpe_fertility":        describe_col("bpe_fertility"),
            "bpe_type_token_ratio": describe_col("bpe_type_token_ratio"),
            "top_30_bpe_subwords":  [
                {"token": t, "count": c}
                for t, c in bpe_freq.most_common(30)
            ],
        }

    # ── Vocabulary extension metrics ──────────────────────────────────────────
    # Corpus-level aggregate (compute over all words + all bpe tokens together)
    vocab_ext_metrics = {}
    if df["vocab_fragmentation_ratio"].notna().any():
        frag_col     = df["vocab_fragmentation_ratio"].dropna()
        ls_col       = df["vocab_long_split_ratio"].dropna()
        cov_col      = df["vocab_tokenizer_coverage"].dropna()

        # Recommendation thresholds
        mean_frag    = frag_col.mean()
        mean_ls      = ls_col.mean()
        mean_cov     = cov_col.mean()

        signals      = []
        if mean_frag > 2.0:
            signals.append(f"Fragmentation ratio {mean_frag:.2f} > 2.0 → high OOV fragmentation")
        if mean_ls > 0.3:
            signals.append(f"Long-split ratio {mean_ls:.2f} > 0.30 → many words split >3 pieces")
        if mean_cov < 0.5:
            signals.append(f"Tokenizer coverage {mean_cov:.2f} < 0.50 → <50% single-token words")

        recommendation = (
            "EXTEND VOCABULARY — strong evidence of poor tokenizer coverage"
            if len(signals) >= 2
            else "CONSIDER extending vocabulary — marginal signal"
            if len(signals) == 1
            else "Vocabulary extension likely NOT required"
        )

        vocab_ext_metrics = {
            "fragmentation_ratio":      describe_col("vocab_fragmentation_ratio"),
            "long_split_ratio":         describe_col("vocab_long_split_ratio"),
            "tokenizer_coverage":       describe_col("vocab_tokenizer_coverage"),
            "threshold_fragmentation":  2.0,
            "threshold_long_split":     0.3,
            "threshold_coverage":       0.5,
            "n_signals_triggered":      len(signals),
            "signals":                  signals,
            "recommendation":           recommendation,
        }

    # ── Category distribution ─────────────────────────────────────────────────
    cat_dist = df["category"].value_counts().to_dict()

    # ── Top words ─────────────────────────────────────────────────────────────
    top_words = [{"word": w, "count": c} for w, c in freq.most_common(50)]

    return {
        "language":             lang,
        "corpus_stats":         corpus_stats,
        "text_metrics":         text_metrics,
        "bpe_stats":            bpe_stats,
        "vocab_extension":      vocab_ext_metrics,
        "category_distribution": cat_dist,
        "top_50_words":         top_words,
    }


def save_lang_json(lang: str, stats: dict):
    """Save per-language stats to eda_outputs/<lang_slug>/stats.json"""
    slug = lang.lower().replace(" ", "-").replace("_", "-")
    lang_dir = OUTPUT_DIR / slug
    lang_dir.mkdir(parents=True, exist_ok=True)
    out_path = lang_dir / "stats.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"  Saved JSON stats: {out_path}")


def save_cross_language_json(all_stats: dict[str, dict]):
    """Save combined cross-language EDA stats to eda_outputs/cross_language/stats.json"""
    cross_dir = OUTPUT_DIR / "cross_language"
    cross_dir.mkdir(parents=True, exist_ok=True)

    # Build comparison table for key metrics
    langs = list(all_stats.keys())

    def get_mean(stats, section, metric):
        try:
            return stats[section][metric]["mean"]
        except (KeyError, TypeError):
            return None

    comparison = {}
    for lang in langs:
        s = all_stats[lang]
        comparison[lang] = {
            "n_documents":           s["corpus_stats"].get("n_documents"),
            "total_tokens":          s["corpus_stats"].get("total_tokens"),
            "vocabulary_size":       s["corpus_stats"].get("vocabulary_size"),
            "mean_word_count":       get_mean(s, "text_metrics", "word_count"),
            "mean_sent_count":       get_mean(s, "text_metrics", "sent_count"),
            "mean_type_token_ratio": get_mean(s, "text_metrics", "type_token_ratio"),
            "mean_avg_word_len":     get_mean(s, "text_metrics", "avg_word_len"),
            "mean_bpe_fertility":    get_mean(s, "bpe_stats", "bpe_fertility") if s.get("bpe_stats") else None,
            "mean_fragmentation_ratio": get_mean(s, "vocab_extension", "fragmentation_ratio"),
            "mean_long_split_ratio":    get_mean(s, "vocab_extension", "long_split_ratio"),
            "mean_tokenizer_coverage":  get_mean(s, "vocab_extension", "tokenizer_coverage"),
            "vocab_recommendation":     s["vocab_extension"].get("recommendation") if s.get("vocab_extension") else None,
        }

    # Vocab overlap (Jaccard) — computed from top_50_words
    jaccard = {}
    vocabs  = {lang: set(w["word"] for w in all_stats[lang].get("top_50_words", [])) for lang in langs}
    for l1 in langs:
        jaccard[l1] = {}
        for l2 in langs:
            inter = len(vocabs[l1] & vocabs[l2])
            union = len(vocabs[l1] | vocabs[l2])
            jaccard[l1][l2] = round(inter / union, 4) if union else 0.0

    combined = {
        "languages":            langs,
        "per_language_summary": comparison,
        "vocabulary_overlap_jaccard": jaccard,
        "full_stats":           all_stats,
    }

    out_path = cross_dir / "stats.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)
    print(f"  Saved cross-language JSON stats: {out_path}")


# ═════════════════════════════════════════════════════════════════════════════
# PLOT HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def save_fig(fig, lang_slug: str, name: str):
    """Save figure into eda_outputs/<lang_slug>/<name>.png"""
    lang_dir = OUTPUT_DIR / lang_slug
    lang_dir.mkdir(parents=True, exist_ok=True)
    out = lang_dir / f"{name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def save_cross_fig(fig, name: str):
    """Save figure into eda_outputs/cross_language/<name>.png"""
    cross_dir = OUTPUT_DIR / "cross_language"
    cross_dir.mkdir(parents=True, exist_ok=True)
    out = cross_dir / f"{name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def stat_box(ax, df: pd.DataFrame, col: str):
    mu  = df[col].mean()
    med = df[col].median()
    ax.axvline(mu,  color="red",    linestyle="--", linewidth=1.2, label=f"Mean={mu:.1f}")
    ax.axvline(med, color="orange", linestyle=":",  linewidth=1.2, label=f"Median={med:.1f}")
    ax.legend()


def _lang_slug(lang: str) -> str:
    return lang.lower().replace(" ", "-").replace("_", "-")


# ═════════════════════════════════════════════════════════════════════════════
# 5. SINGLE-LANGUAGE EDA
# ═════════════════════════════════════════════════════════════════════════════

def plot_single_language(lang: str, df: pd.DataFrame):
    print(f"\n[{lang}] Generating single-language plots …")
    slug = _lang_slug(lang)

    # ── 4-A. Univariate distributions ────────────────────────────────────────
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.suptitle(f"{lang} Corpus — Univariate Distribution Overview", fontsize=15, fontweight="bold", y=1.01)

    numeric_cols = [
        ("word_count",      "Word Count per Document"),
        ("unique_words",    "Unique Words per Document"),
        ("char_count",      "Character Count per Document"),
        ("sent_count",      "Sentence Count per Document"),
        ("avg_sent_len",    "Avg Sentence Length (words)"),
        ("avg_word_len",    "Avg Word Length (chars)"),
        ("type_token_ratio","Type-Token Ratio (TTR)"),
        ("para_count",      "Paragraph Count per Document"),
    ]

    for ax, (col, title) in zip(axes.flat, numeric_cols):
        data = df[col].dropna()
        ax.hist(data, bins=min(30, max(5, len(data)//3)), color=PALETTE[0], edgecolor="white", linewidth=0.5)
        stat_box(ax, df, col)
        ax.set_title(title)
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")

    fig.tight_layout()
    save_fig(fig, slug, "01_univariate")

    # ── 4-B. Correlation heatmap ──────────────────────────────────────────────
    num_df = df[["word_count","unique_words","char_count","sent_count",
                 "avg_sent_len","avg_word_len","type_token_ratio","para_count"]].dropna()

    if len(num_df) > 1:
        fig, ax = plt.subplots(figsize=(9, 7))
        corr = num_df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                    center=0, linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
        ax.set_title(f"{lang} — Feature Correlation Heatmap")
        fig.tight_layout()
        save_fig(fig, slug, "02_correlation")

    # ── 4-C. Category distribution ────────────────────────────────────────────
    if df["category"].nunique() > 1:
        cat_counts = df["category"].value_counts()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"{lang} — Category Distribution", fontsize=13, fontweight="bold")

        ax1.barh(cat_counts.index, cat_counts.values, color=PALETTE[:len(cat_counts)])
        ax1.set_xlabel("Document Count")
        ax1.set_title("Documents per Category")
        for i, v in enumerate(cat_counts.values):
            ax1.text(v + 0.1, i, str(v), va="center", fontsize=8)

        ax2.pie(cat_counts.values, labels=cat_counts.index, autopct="%1.1f%%",
                colors=PALETTE[:len(cat_counts)], startangle=140)
        ax2.set_title("Category Share (%)")

        fig.tight_layout()
        save_fig(fig, slug, "03_category_dist")

    # ── 4-D. Word count vs TTR scatter ────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{lang} — Vocabulary Richness", fontsize=13, fontweight="bold")

    axes[0].scatter(df["word_count"], df["type_token_ratio"],
                    alpha=0.6, color=PALETTE[1], edgecolors="white", s=40)
    axes[0].set_xlabel("Word Count")
    axes[0].set_ylabel("Type-Token Ratio (TTR)")
    axes[0].set_title("Word Count vs. TTR")

    axes[1].scatter(df["word_count"], df["unique_words"],
                    alpha=0.6, color=PALETTE[2], edgecolors="white", s=40)
    x_vals = np.linspace(df["word_count"].min(), df["word_count"].max(), 200)
    axes[1].plot(x_vals, x_vals**0.67 * (df["unique_words"].max() / df["word_count"].max()**0.67),
                 "r--", linewidth=1, label="Heaps' law ref (~N^0.67)")
    axes[1].set_xlabel("Word Count (Tokens)")
    axes[1].set_ylabel("Unique Words (Types)")
    axes[1].set_title("Token–Type Growth")
    axes[1].legend()

    fig.tight_layout()
    save_fig(fig, slug, "04_vocabulary_richness")

    # ── 4-E. Top-N word frequency ─────────────────────────────────────────────
    all_words = [w for words in df["_words"] for w in words]
    freq = Counter(all_words)
    top_n = 30
    top_words = freq.most_common(top_n)

    fig, ax = plt.subplots(figsize=(14, 6))
    words_x, counts_y = zip(*top_words)
    bars = ax.bar(words_x, counts_y, color=PALETTE[3], edgecolor="white")
    ax.set_title(f"{lang} — Top {top_n} Most Frequent Words")
    ax.set_xlabel("Word")
    ax.set_ylabel("Frequency")
    ax.set_xticklabels(words_x, rotation=45, ha="right")
    for bar, cnt in zip(bars, counts_y):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts_y)*0.005,
                str(cnt), ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    save_fig(fig, slug, "05_top_words")

    # ── 4-F. Zipf's law ──────────────────────────────────────────────────────
    ranks  = np.arange(1, len(freq) + 1)
    counts = np.array([c for _, c in freq.most_common()])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.loglog(ranks, counts, "o", markersize=2, alpha=0.5, color=PALETTE[4], label="Observed")
    zipf_ref = counts[0] / ranks
    ax.loglog(ranks, zipf_ref, "r--", linewidth=1.5, label="Zipf (α=1) reference")
    ax.set_xlabel("Rank (log scale)")
    ax.set_ylabel("Frequency (log scale)")
    ax.set_title(f"{lang} — Zipf's Law Analysis")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, slug, "06_zipf_law")

    # ── 4-G. Word Cloud ───────────────────────────────────────────────────────
    try:
        text_blob = " ".join(all_words)
        wc = WordCloud(width=1200, height=600, background_color="white",
                       max_words=200, colormap="tab10",
                       prefer_horizontal=0.8).generate(text_blob)
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(f"{lang} — Word Cloud (top 200 words)", fontsize=14, fontweight="bold")
        fig.tight_layout()
        save_fig(fig, slug, "07_wordcloud")
    except Exception as e:
        print(f"  [WARN] WordCloud skipped: {e}")

    # ── 4-H. Document length box plots ───────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"{lang} — Document Length Box Plots", fontsize=13, fontweight="bold")

    for ax, (col, label) in zip(axes, [
        ("word_count",  "Words"),
        ("char_count",  "Characters"),
        ("sent_count",  "Sentences"),
    ]):
        bp = ax.boxplot(df[col].dropna(), patch_artist=True, notch=False,
                        boxprops=dict(facecolor=PALETTE[5], alpha=0.7),
                        medianprops=dict(color="red", linewidth=2))
        ax.set_title(f"{label} per Document")
        ax.set_ylabel(label)
        ax.set_xticks([])
        q1, med, q3 = df[col].quantile([0.25, 0.5, 0.75])
        ax.text(1.35, med, f"Med={med:.0f}", va="center", fontsize=8, color="red")
        ax.text(1.35, q3,  f"Q3={q3:.0f}",  va="center", fontsize=8)
        ax.text(1.35, q1,  f"Q1={q1:.0f}",  va="center", fontsize=8)

    fig.tight_layout()
    save_fig(fig, slug, "08_boxplots")

    # ── 4-I. Sentence count per document ─────────────────────────────────────
    script_label = df["script"].mode()[0].upper() if "script" in df.columns else "LATIN"
    sent_counts  = df["sent_count"].dropna().astype(int)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle(
        f"{lang} — Sentence Count per Document\n"
        f"(Script: {script_label} | "
        f"{'Khmer terminator ។ / ៕' if script_label == 'KHMER' else 'Latin terminators . ! ? …'})",
        fontsize=13, fontweight="bold"
    )

    ax = axes[0]
    n_bins = min(40, max(5, len(sent_counts) // 3))
    ax.hist(sent_counts, bins=n_bins, color=PALETTE[6], edgecolor="white", linewidth=0.5)
    ax.axvline(sent_counts.mean(),   color="red",    linestyle="--", linewidth=1.4,
               label=f"Mean = {sent_counts.mean():.1f}")
    ax.axvline(sent_counts.median(), color="orange", linestyle=":",  linewidth=1.4,
               label=f"Median = {sent_counts.median():.1f}")
    ax.set_xlabel("Sentence Count")
    ax.set_ylabel("Number of Documents")
    ax.set_title("Histogram of Sentence Counts")
    ax.legend()

    ax = axes[1]
    sample = sent_counts.sort_values(ascending=False).reset_index(drop=True)
    if len(sample) > 200:
        sample = sample.iloc[:200]
        ax.set_title(f"Sentence Count per Doc\n(top 200 of {len(sent_counts)} shown)")
    else:
        ax.set_title("Sentence Count per Document\n(sorted descending)")
    bar_colors = [PALETTE[6] if v >= sent_counts.mean() else PALETTE[7] for v in sample]
    ax.bar(range(len(sample)), sample.values, color=bar_colors, width=1.0, edgecolor="none")
    ax.axhline(sent_counts.mean(),   color="red",    linestyle="--", linewidth=1.2,
               label=f"Mean={sent_counts.mean():.1f}")
    ax.axhline(sent_counts.median(), color="orange", linestyle=":",  linewidth=1.2,
               label=f"Median={sent_counts.median():.1f}")
    ax.set_xlabel("Document Index (sorted)")
    ax.set_ylabel("Sentence Count")
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.axis("off")
    stats = {
        "Total documents":    f"{len(sent_counts):,}",
        "Total sentences":    f"{sent_counts.sum():,}",
        "Mean  sent/doc":     f"{sent_counts.mean():.2f}",
        "Median sent/doc":    f"{sent_counts.median():.2f}",
        "Std  sent/doc":      f"{sent_counts.std():.2f}",
        "Min  sent/doc":      f"{sent_counts.min()}",
        "Max  sent/doc":      f"{sent_counts.max()}",
        "Q1  (25%)":          f"{sent_counts.quantile(0.25):.1f}",
        "Q3  (75%)":          f"{sent_counts.quantile(0.75):.1f}",
        "Docs with 1 sent":   f"{(sent_counts == 1).sum()} ({100*(sent_counts==1).mean():.1f}%)",
        "Script type":        script_label,
        "Sentence splitter":  "Khmer ។/៕" if script_label == "KHMER" else "Latin .!?…",
    }
    cell_text  = [[k, v] for k, v in stats.items()]
    tbl = ax.table(cellText=cell_text, colLabels=["Metric", "Value"],
                   cellLoc="left", loc="center", bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(GRID_COLOR)
        if r == 0:
            cell.set_facecolor(PALETTE[6])
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#EEF2FF")
        else:
            cell.set_facecolor("white")
    ax.set_title("Summary Statistics", fontweight="bold", pad=10)

    fig.tight_layout()
    save_fig(fig, slug, "09_sentence_counts")

    # ── 4-J. BPE Token Analysis ───────────────────────────────────────────────
    if df["bpe_token_count"].notna().any():
        bpe_counts  = df["bpe_token_count"].dropna()
        fertility   = df["bpe_fertility"].dropna()
        bpe_ttr     = df["bpe_type_token_ratio"].dropna()

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(
            f"{lang} — XLM-RoBERTa BPE Token Analysis\n"
            f"(model: xlm-roberta-base  |  special tokens excluded)",
            fontsize=13, fontweight="bold"
        )

        ax = axes[0, 0]
        ax.hist(bpe_counts, bins=min(40, max(5, len(bpe_counts)//3)),
                color=PALETTE[0], edgecolor="white")
        ax.axvline(bpe_counts.mean(),   color="red",    linestyle="--", lw=1.4,
                   label=f"Mean={bpe_counts.mean():.1f}")
        ax.axvline(bpe_counts.median(), color="orange", linestyle=":",  lw=1.4,
                   label=f"Median={bpe_counts.median():.1f}")
        ax.set_title("BPE Token Count per Document")
        ax.set_xlabel("BPE Token Count")
        ax.set_ylabel("Frequency")
        ax.legend()

        ax = axes[0, 1]
        ax.scatter(df["word_count"].dropna(), bpe_counts,
                   alpha=0.5, color=PALETTE[1], edgecolors="none", s=25)
        lim = max(df["word_count"].max(), bpe_counts.max())
        ax.plot([0, lim], [0, lim], "k--", lw=0.8, label="1:1 (no fragmentation)")
        ax.set_title("Whitespace Words vs BPE Tokens")
        ax.set_xlabel("Whitespace Word Count")
        ax.set_ylabel("BPE Token Count")
        ax.legend(fontsize=8)

        ax = axes[0, 2]
        ax.hist(fertility, bins=min(40, max(5, len(fertility)//3)),
                color=PALETTE[2], edgecolor="white")
        ax.axvline(fertility.mean(),   color="red",    linestyle="--", lw=1.4,
                   label=f"Mean={fertility.mean():.2f}")
        ax.axvline(1.0,                color="black",  linestyle=":",  lw=1.0,
                   label="Fertility=1 (no split)")
        ax.set_title("BPE Fertility per Document")
        ax.set_xlabel("Fertility")
        ax.set_ylabel("Frequency")
        ax.legend()

        ax = axes[1, 0]
        all_bpe = [t for toks in df["_bpe_tokens"] for t in toks]
        bpe_freq = Counter(all_bpe)
        top_bpe  = bpe_freq.most_common(30)
        bpe_words, bpe_cnts = zip(*top_bpe)
        bars = ax.bar(bpe_words, bpe_cnts, color=PALETTE[3], edgecolor="white")
        ax.set_title("Top 30 Most Frequent BPE Subwords")
        ax.set_xlabel("Subword Token")
        ax.set_ylabel("Frequency")
        ax.set_xticklabels(bpe_words, rotation=50, ha="right", fontsize=7)
        for bar, cnt in zip(bars, bpe_cnts):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(bpe_cnts)*0.005,
                    str(cnt), ha="center", fontsize=6)

        ax = axes[1, 1]
        ax.scatter(df["type_token_ratio"].dropna(), bpe_ttr,
                   alpha=0.5, color=PALETTE[4], edgecolors="none", s=25)
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Equal TTR")
        ax.set_title("Whitespace TTR vs BPE TTR")
        ax.set_xlabel("Whitespace Type-Token Ratio")
        ax.set_ylabel("BPE Type-Token Ratio")
        ax.legend(fontsize=8)

        ax = axes[1, 2]
        ax.axis("off")
        stats = {
            "Total BPE tokens":      f"{int(bpe_counts.sum()):,}",
            "Mean BPE/doc":          f"{bpe_counts.mean():.1f}",
            "Median BPE/doc":        f"{bpe_counts.median():.1f}",
            "Std BPE/doc":           f"{bpe_counts.std():.1f}",
            "Min BPE/doc":           f"{int(bpe_counts.min())}",
            "Max BPE/doc":           f"{int(bpe_counts.max())}",
            "Mean fertility":        f"{fertility.mean():.3f}",
            "Median fertility":      f"{fertility.median():.3f}",
            "Unique BPE vocab":      f"{len(bpe_freq):,}",
            "Top subword":           f"{top_bpe[0][0]} ({top_bpe[0][1]:,}×)",
            "BPE model":             "xlm-roberta-base",
            "Special tokens":        "excluded (<s> </s>)",
        }
        cell_text = [[k, v] for k, v in stats.items()]
        tbl = ax.table(cellText=cell_text, colLabels=["Metric", "Value"],
                       cellLoc="left", loc="center", bbox=[0, 0, 1, 1])
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8.5)
        for (r, c), cell in tbl.get_celld().items():
            cell.set_edgecolor(GRID_COLOR)
            if r == 0:
                cell.set_facecolor(PALETTE[0])
                cell.set_text_props(color="white", fontweight="bold")
            elif r % 2 == 0:
                cell.set_facecolor("#EEF7FF")
            else:
                cell.set_facecolor("white")
        ax.set_title("BPE Summary Statistics", fontweight="bold", pad=10)

        fig.tight_layout()
        save_fig(fig, slug, "10_bpe_analysis")

    # ── 4-K. Vocabulary Extension Assessment (NEW) ────────────────────────────
    if df["vocab_fragmentation_ratio"].notna().any():
        plot_vocab_extension_single(lang, df, slug)

    print(f"[{lang}] ✓ All single-language plots done.")


# ═════════════════════════════════════════════════════════════════════════════
# 4-K. VOCABULARY EXTENSION ASSESSMENT — SINGLE LANGUAGE
# ═════════════════════════════════════════════════════════════════════════════

# Threshold reference lines
_THRESH_FRAG = 2.0    # fragmentation_ratio > 2.0 → concern
_THRESH_LS   = 0.30   # long_split_ratio    > 0.30 → concern
_THRESH_COV  = 0.50   # tokenizer_coverage  < 0.50 → concern


def _add_threshold_annotation(ax, value, threshold, is_above_bad: bool,
                               label: str, unit: str = ""):
    """Draw threshold line + pass/fail badge on an axes."""
    ax.axhline(threshold, color="#E74C3C", linestyle="--", linewidth=1.5,
               label=f"Threshold = {threshold}{unit}")

    bad = (value > threshold) if is_above_bad else (value < threshold)
    badge_color = "#E74C3C" if bad else "#27AE60"
    badge_text  = "⚠ EXCEEDS THRESHOLD" if bad else "✓ WITHIN THRESHOLD"

    ax.annotate(badge_text,
                xy=(0.98, 0.92), xycoords="axes fraction",
                ha="right", va="top", fontsize=8.5, fontweight="bold",
                color="white",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=badge_color, alpha=0.9))


def plot_vocab_extension_single(lang: str, df: pd.DataFrame, slug: str):
    """
    Plot 11 — Vocabulary Extension Assessment for a single language.

    Layout  (2 rows × 3 cols):
    ┌─────────────────────┬─────────────────────┬──────────────────────┐
    │ Hist: Frag Ratio    │ Hist: Long-Split     │ Hist: Cov            │
    │ (doc-level distrib) │ Ratio                │                      │
    ├─────────────────────┼─────────────────────┼──────────────────────┤
    │ Gauge: mean frag    │ Gauge: mean long-spl │ Summary / Verdict    │
    └─────────────────────┴─────────────────────┴──────────────────────┘
    """
    frag_col = df["vocab_fragmentation_ratio"].dropna()
    ls_col   = df["vocab_long_split_ratio"].dropna()
    cov_col  = df["vocab_tokenizer_coverage"].dropna()

    mean_frag = frag_col.mean()
    mean_ls   = ls_col.mean()
    mean_cov  = cov_col.mean()

    signals = []
    if mean_frag > _THRESH_FRAG:
        signals.append("Fragmentation Ratio")
    if mean_ls > _THRESH_LS:
        signals.append("Long-Split Ratio")
    if mean_cov < _THRESH_COV:
        signals.append("Tokenizer Coverage")

    verdict_color = "#E74C3C" if len(signals) >= 2 else "#F39C12" if len(signals) == 1 else "#27AE60"
    verdict_text  = (
        "EXTEND VOCABULARY" if len(signals) >= 2
        else "CONSIDER EXTENDING" if len(signals) == 1
        else "NO EXTENSION NEEDED"
    )

    fig = plt.figure(figsize=(18, 11))
    fig.patch.set_facecolor(BG_COLOR)
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── Title banner ──────────────────────────────────────────────────────────
    fig.suptitle(
        f"{lang} — Vocabulary Extension Assessment\n"
        f"(XLM-RoBERTa xlm-roberta-base  |  3 diagnostic metrics)",
        fontsize=14, fontweight="bold", y=1.01,
    )

    # ── Row 0: histograms ─────────────────────────────────────────────────────

    # (0,0)  Fragmentation Ratio histogram
    ax00 = fig.add_subplot(gs[0, 0])
    n_bins = min(40, max(5, len(frag_col) // 3))
    ax00.hist(frag_col, bins=n_bins, color=PALETTE[0], edgecolor="white", linewidth=0.5)
    ax00.axvline(mean_frag, color="red", linestyle="--", linewidth=1.5,
                 label=f"Mean = {mean_frag:.2f}")
    ax00.axvline(_THRESH_FRAG, color="#E74C3C", linestyle=":", linewidth=1.5,
                 label=f"Threshold = {_THRESH_FRAG}")
    ax00.set_title("① Fragmentation Ratio\n(BPE tokens ÷ whitespace words)", fontweight="bold")
    ax00.set_xlabel("Fragmentation Ratio per Document")
    ax00.set_ylabel("Number of Documents")
    ax00.legend(fontsize=8)
    _shade_bad_region(ax00, frag_col, _THRESH_FRAG, above=True)

    # (0,1)  Long-Split Ratio histogram
    ax01 = fig.add_subplot(gs[0, 1])
    n_bins = min(40, max(5, len(ls_col) // 3))
    ax01.hist(ls_col, bins=n_bins, color=PALETTE[2], edgecolor="white", linewidth=0.5)
    ax01.axvline(mean_ls, color="red", linestyle="--", linewidth=1.5,
                 label=f"Mean = {mean_ls:.3f}")
    ax01.axvline(_THRESH_LS, color="#E74C3C", linestyle=":", linewidth=1.5,
                 label=f"Threshold = {_THRESH_LS}")
    ax01.set_title("② Long-Split Ratio\n(fraction of words split into > 3 pieces)", fontweight="bold")
    ax01.set_xlabel("Long-Split Ratio per Document")
    ax01.set_ylabel("Number of Documents")
    ax01.legend(fontsize=8)
    _shade_bad_region(ax01, ls_col, _THRESH_LS, above=True)

    # (0,2)  Tokenizer Coverage histogram
    ax02 = fig.add_subplot(gs[0, 2])
    n_bins = min(40, max(5, len(cov_col) // 3))
    ax02.hist(cov_col, bins=n_bins, color=PALETTE[4], edgecolor="white", linewidth=0.5)
    ax02.axvline(mean_cov, color="red", linestyle="--", linewidth=1.5,
                 label=f"Mean = {mean_cov:.3f}")
    ax02.axvline(_THRESH_COV, color="#E74C3C", linestyle=":", linewidth=1.5,
                 label=f"Threshold = {_THRESH_COV}")
    ax02.set_title("③ Tokenizer Coverage\n(fraction of words → single BPE token)", fontweight="bold")
    ax02.set_xlabel("Coverage per Document")
    ax02.set_ylabel("Number of Documents")
    ax02.legend(fontsize=8)
    _shade_bad_region(ax02, cov_col, _THRESH_COV, above=False)

    # ── Row 1: gauge bars + summary ───────────────────────────────────────────

    # (1,0)  Gauge: Fragmentation Ratio
    ax10 = fig.add_subplot(gs[1, 0])
    _draw_gauge_bar(
        ax10,
        value=mean_frag, threshold=_THRESH_FRAG,
        is_above_bad=True,
        label="Mean Fragmentation Ratio",
        color=PALETTE[0],
        x_max=max(_THRESH_FRAG * 2, mean_frag * 1.2),
    )

    # (1,1)  Gauge: Long-Split Ratio + Coverage side-by-side
    ax11 = fig.add_subplot(gs[1, 1])
    _draw_dual_gauge(
        ax11,
        val_a=mean_ls,   thresh_a=_THRESH_LS,  label_a="Long-Split Ratio",  above_bad_a=True,
        val_b=mean_cov,  thresh_b=_THRESH_COV, label_b="Tokenizer Coverage", above_bad_b=False,
    )

    # (1,2)  Summary / Verdict table
    ax12 = fig.add_subplot(gs[1, 2])
    ax12.axis("off")

    # Verdict banner
    fancy = FancyBboxPatch(
        (0.05, 0.72), 0.90, 0.22,
        boxstyle="round,pad=0.02",
        linewidth=2, edgecolor=verdict_color,
        facecolor=verdict_color, transform=ax12.transAxes, clip_on=False,
    )
    ax12.add_patch(fancy)
    ax12.text(0.50, 0.83, verdict_text,
              transform=ax12.transAxes,
              ha="center", va="center",
              fontsize=13, fontweight="bold", color="white")
    ax12.text(0.50, 0.75,
              f"({len(signals)}/3 thresholds exceeded)",
              transform=ax12.transAxes,
              ha="center", va="center", fontsize=9, color="white")

    # Metric summary table
    rows_data = [
        ["Metric",                "Mean",               f"Threshold", "Status"],
        ["Fragmentation Ratio",   f"{mean_frag:.3f}",   f"> {_THRESH_FRAG}",
         "⚠ FAIL" if mean_frag > _THRESH_FRAG else "✓ OK"],
        ["Long-Split Ratio",      f"{mean_ls:.3f}",     f"> {_THRESH_LS}",
         "⚠ FAIL" if mean_ls > _THRESH_LS else "✓ OK"],
        ["Tokenizer Coverage",    f"{mean_cov:.3f}",    f"< {_THRESH_COV}",
         "⚠ FAIL" if mean_cov < _THRESH_COV else "✓ OK"],
    ]

    tbl = ax12.table(
        cellText=rows_data[1:],
        colLabels=rows_data[0],
        cellLoc="center", loc="center",
        bbox=[0.0, 0.0, 1.0, 0.68],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(GRID_COLOR)
        if r == 0:
            cell.set_facecolor("#2C3E50")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            status = rows_data[r][3]
            if c == 3:
                cell.set_facecolor("#FADBD8" if "FAIL" in status else "#D5F5E3")
                cell.set_text_props(
                    color="#C0392B" if "FAIL" in status else "#1E8449",
                    fontweight="bold"
                )
            elif r % 2 == 0:
                cell.set_facecolor("#F2F3F4")
            else:
                cell.set_facecolor("white")
    ax12.set_title("Verdict & Metric Summary", fontweight="bold", pad=10)

    save_fig(fig, slug, "11_vocab_extension_assessment")


def _shade_bad_region(ax, data, threshold, above: bool):
    """Shade the 'bad' region of a histogram in light red."""
    xlim = ax.get_xlim()
    if above:
        ax.axvspan(threshold, max(xlim[1], data.max() * 1.05),
                   alpha=0.08, color="#E74C3C", label="_nolegend_")
    else:
        ax.axvspan(min(xlim[0], 0), threshold,
                   alpha=0.08, color="#E74C3C", label="_nolegend_")


def _draw_gauge_bar(ax, value, threshold, is_above_bad: bool,
                    label: str, color, x_max: float):
    """Draw a horizontal gauge bar showing value vs threshold."""
    bad = (value > threshold) if is_above_bad else (value < threshold)
    bar_color = "#E74C3C" if bad else "#27AE60"

    ax.set_xlim(0, x_max)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel(label)
    ax.set_title(f"{label}\nMean = {value:.3f}  |  Threshold = {threshold}", fontweight="bold")

    # Background bar
    ax.barh(0.5, x_max, height=0.4, color="#ECF0F1", left=0)
    # Value bar
    ax.barh(0.5, value, height=0.4, color=bar_color, left=0,
            label=f"Mean = {value:.3f}")
    # Threshold line
    ax.axvline(threshold, color="#E74C3C", linewidth=2, linestyle="--",
               label=f"Threshold = {threshold}")
    # Value label
    ax.text(value + x_max * 0.01, 0.5, f"{value:.3f}",
            va="center", fontsize=10, fontweight="bold", color=bar_color)

    badge = "⚠ FAIL" if bad else "✓ PASS"
    ax.text(x_max * 0.98, 0.5, badge,
            ha="right", va="center", fontsize=11, fontweight="bold",
            color="white",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=bar_color, alpha=0.9))
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", visible=False)


def _draw_dual_gauge(ax,
                     val_a, thresh_a, label_a, above_bad_a,
                     val_b, thresh_b, label_b, above_bad_b):
    """Draw two horizontal gauge bars stacked vertically."""
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.1, 2.1)
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels([label_b, label_a], fontsize=8)
    ax.set_xlabel("Value (0 – 1 scale)")
    ax.set_title("Long-Split Ratio  &  Tokenizer Coverage\n(normalised to [0,1])", fontweight="bold")

    for y_pos, val, thresh, is_above in [
        (1.5, val_a, thresh_a, above_bad_a),
        (0.5, val_b, thresh_b, above_bad_b),
    ]:
        bad = (val > thresh) if is_above else (val < thresh)
        bc  = "#E74C3C" if bad else "#27AE60"
        ax.barh(y_pos, 1.0, height=0.35, color="#ECF0F1", left=0)
        ax.barh(y_pos, min(val, 1.0), height=0.35, color=bc, left=0)
        ax.axvline(thresh, color="#E74C3C", linewidth=1.5, linestyle="--")
        ax.text(min(val, 1.0) + 0.01, y_pos, f"{val:.3f}",
                va="center", fontsize=9, fontweight="bold", color=bc)
        badge = "⚠ FAIL" if bad else "✓ PASS"
        ax.text(0.99, y_pos, badge, ha="right", va="center",
                fontsize=9, fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.2", facecolor=bc, alpha=0.9))

    ax.grid(axis="x", color=GRID_COLOR, linestyle="--", linewidth=0.6)
    ax.grid(axis="y", visible=False)


# ═════════════════════════════════════════════════════════════════════════════
# 6. CROSS-LANGUAGE COMPARISON
# ═════════════════════════════════════════════════════════════════════════════

def plot_cross_language(all_dfs: dict[str, pd.DataFrame]):
    if len(all_dfs) < 2:
        print("  Skipping cross-language plots (only 1 language loaded).")
        return

    print("\n[Cross-Language] Generating comparison plots …")

    langs  = list(all_dfs.keys())
    colors = {lang: PALETTE[i] for i, lang in enumerate(langs)}

    # ── 5-A. Key metrics box plots ────────────────────────────────────────────
    metrics = [
        ("word_count",       "Word Count"),
        ("unique_words",     "Unique Words"),
        ("type_token_ratio", "Type-Token Ratio"),
        ("avg_sent_len",     "Avg Sentence Length"),
        ("avg_word_len",     "Avg Word Length"),
        ("sent_count",       "Sentence Count"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Cross-Language Corpus Comparison — Key Metrics", fontsize=15, fontweight="bold")

    for ax, (col, label) in zip(axes.flat, metrics):
        data   = [all_dfs[l][col].dropna().values for l in langs]
        bp = ax.boxplot(data, patch_artist=True, notch=False,
                        labels=langs, showfliers=True)
        for patch, lang in zip(bp["boxes"], langs):
            patch.set_facecolor(colors[lang])
            patch.set_alpha(0.7)
        for median in bp["medians"]:
            median.set_color("red")
            median.set_linewidth(2)
        ax.set_title(label)
        ax.set_ylabel(label)
        ax.tick_params(axis="x", rotation=15)

    fig.tight_layout()
    save_cross_fig(fig, "01_comparison_boxplots")

    # ── 5-B. Corpus size ──────────────────────────────────────────────────────
    summary = {
        lang: {
            "Documents":      len(df),
            "Total Words":    int(df["word_count"].sum()),
            "Unique Vocab":   int(pd.Series([w for ws in df["_words"] for w in ws]).nunique()),
            "Avg Doc Length": float(df["word_count"].mean()),
        }
        for lang, df in all_dfs.items()
    }
    summary_df = pd.DataFrame(summary).T

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle("Cross-Language Corpus Overview", fontsize=14, fontweight="bold")

    for ax, col in zip(axes, summary_df.columns):
        vals  = summary_df[col]
        bars  = ax.bar(langs, vals, color=[colors[l] for l in langs], edgecolor="white")
        ax.set_title(col)
        ax.set_ylabel(col)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01,
                    f"{v:,.0f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    save_cross_fig(fig, "02_corpus_size")

    # ── 5-C. Jaccard similarity ───────────────────────────────────────────────
    vocab = {lang: set(w for ws in df["_words"] for w in ws) for lang, df in all_dfs.items()}
    jaccard = pd.DataFrame(index=langs, columns=langs, dtype=float)
    for l1 in langs:
        for l2 in langs:
            inter = len(vocab[l1] & vocab[l2])
            union = len(vocab[l1] | vocab[l2])
            jaccard.loc[l1, l2] = inter / union if union else 0.0

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(jaccard.astype(float), annot=True, fmt=".3f", cmap="YlOrRd",
                linewidths=0.5, ax=ax, vmin=0, vmax=1)
    ax.set_title("Vocabulary Overlap — Jaccard Similarity")
    fig.tight_layout()
    save_cross_fig(fig, "03_vocab_jaccard")

    # ── 5-D. KDE overlay ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Cross-Language Distribution Overlay", fontsize=13, fontweight="bold")

    for ax, (col, label) in zip(axes, [("word_count","Word Count"), ("type_token_ratio","Type-Token Ratio")]):
        for lang, df in all_dfs.items():
            data = df[col].dropna()
            if len(data) > 1:
                sns.kdeplot(data, ax=ax, label=lang, color=colors[lang], linewidth=2, fill=True, alpha=0.15)
            else:
                ax.axvline(data.iloc[0], label=lang, color=colors[lang], linewidth=2)
        ax.set_xlabel(label)
        ax.set_ylabel("Density")
        ax.set_title(f"Distribution of {label}")
        ax.legend()

    fig.tight_layout()
    save_cross_fig(fig, "04_kde_overlay")

    # ── 5-E. Word freq heatmap ────────────────────────────────────────────────
    top_per_lang = {}
    for lang, df in all_dfs.items():
        all_words = [w for ws in df["_words"] for w in ws]
        top_per_lang[lang] = dict(Counter(all_words).most_common(50))

    all_top_words = list({w for d in top_per_lang.values() for w in list(d.keys())[:20]})
    heat_data = pd.DataFrame(
        {lang: [top_per_lang[lang].get(w, 0) for w in all_top_words] for lang in langs},
        index=all_top_words
    )
    heat_norm = heat_data.div(heat_data.max()).fillna(0)

    fig, ax = plt.subplots(figsize=(max(8, len(langs)*2), max(10, len(all_top_words)*0.4)))
    sns.heatmap(heat_norm, cmap="Blues", linewidths=0.3, ax=ax,
                cbar_kws={"label": "Normalized Frequency"})
    ax.set_title("Top Word Frequency Heatmap Across Languages\n(normalized per language)")
    fig.tight_layout()
    save_cross_fig(fig, "05_word_freq_heatmap")

    # ── 5-F. Sentence count comparison ───────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Cross-Language Sentence Count Comparison", fontsize=13, fontweight="bold")

    ax = axes[0]
    data_lists = [all_dfs[l]["sent_count"].dropna().values for l in langs]
    bp = ax.boxplot(data_lists, patch_artist=True, labels=langs, showfliers=True)
    for patch, lang in zip(bp["boxes"], langs):
        patch.set_facecolor(colors[lang]); patch.set_alpha(0.7)
    for median in bp["medians"]:
        median.set_color("red"); median.set_linewidth(2)
    ax.set_title("Sentence Count Distribution per Language")
    ax.set_ylabel("Sentence Count per Document")
    ax.tick_params(axis="x", rotation=15)

    ax = axes[1]
    means  = [all_dfs[l]["sent_count"].mean()  for l in langs]
    stds   = [all_dfs[l]["sent_count"].std()   for l in langs]
    bars   = ax.bar(langs, means, yerr=stds, color=[colors[l] for l in langs],
                    edgecolor="white", capsize=6, error_kw={"linewidth": 1.5})
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + s + max(means)*0.01,
                f"{m:.1f}±{s:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_title("Mean Sentence Count per Doc (± 1 std)")
    ax.set_ylabel("Mean Sentence Count")
    ax.tick_params(axis="x", rotation=15)

    ax = axes[2]
    for lang in langs:
        data   = all_dfs[lang]["sent_count"].dropna()
        script = all_dfs[lang]["script"].mode()[0] if "script" in all_dfs[lang].columns else "latin"
        note   = "Khmer ។/៕" if script == "khmer" else "Latin .!?…"
        if len(data) > 1:
            sns.kdeplot(data, ax=ax, label=f"{lang} ({note})",
                        color=colors[lang], linewidth=2, fill=True, alpha=0.15)
        else:
            ax.axvline(data.iloc[0], color=colors[lang], linewidth=2, label=f"{lang} (n=1)")
    ax.set_xlabel("Sentence Count per Document")
    ax.set_ylabel("Density")
    ax.set_title("KDE of Sentence Counts")
    ax.legend()

    fig.tight_layout()
    save_cross_fig(fig, "06_sentence_count_comparison")

    # ── 5-G. BPE cross-language ───────────────────────────────────────────────
    has_bpe = {l: all_dfs[l]["bpe_token_count"].notna().any() for l in langs}
    if any(has_bpe.values()):
        bpe_langs = [l for l in langs if has_bpe[l]]

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(
            "Cross-Language BPE Token Analysis — XLM-RoBERTa (xlm-roberta-base)",
            fontsize=13, fontweight="bold"
        )

        ax = axes[0]
        data_bpe = [all_dfs[l]["bpe_token_count"].dropna().values for l in bpe_langs]
        bp = ax.boxplot(data_bpe, patch_artist=True, labels=bpe_langs, showfliers=True)
        for patch, l in zip(bp["boxes"], bpe_langs):
            patch.set_facecolor(colors[l]); patch.set_alpha(0.7)
        for med in bp["medians"]:
            med.set_color("red"); med.set_linewidth(2)
        ax.set_title("BPE Token Count per Document")
        ax.set_ylabel("BPE Token Count")
        ax.tick_params(axis="x", rotation=15)

        ax = axes[1]
        data_fert = [all_dfs[l]["bpe_fertility"].dropna().values for l in bpe_langs]
        bp2 = ax.boxplot(data_fert, patch_artist=True, labels=bpe_langs, showfliers=True)
        for patch, l in zip(bp2["boxes"], bpe_langs):
            patch.set_facecolor(colors[l]); patch.set_alpha(0.7)
        for med in bp2["medians"]:
            med.set_color("red"); med.set_linewidth(2)
        ax.axhline(1.0, color="black", linestyle="--", lw=1.0, label="Fertility=1")
        ax.set_title("BPE Fertility per Language")
        ax.set_ylabel("Fertility (BPE / whitespace)")
        ax.legend(fontsize=8)
        ax.tick_params(axis="x", rotation=15)

        ax = axes[2]
        x      = np.arange(len(bpe_langs))
        width  = 0.35
        means_ws  = [all_dfs[l]["word_count"].mean()     for l in bpe_langs]
        means_bpe = [all_dfs[l]["bpe_token_count"].mean() for l in bpe_langs]
        b1 = ax.bar(x - width/2, means_ws,  width, label="Whitespace words",
                    color=[colors[l] for l in bpe_langs], alpha=0.6, edgecolor="white")
        b2 = ax.bar(x + width/2, means_bpe, width, label="BPE tokens",
                    color=[colors[l] for l in bpe_langs], alpha=1.0, edgecolor="white", hatch="//")
        for bars in [b1, b2]:
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + max(means_bpe)*0.01,
                        f"{bar.get_height():.0f}",
                        ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(bpe_langs, rotation=15)
        ax.set_title("Mean Whitespace Words vs BPE Tokens per Document")
        ax.set_ylabel("Mean Count per Document")
        ax.legend()

        fig.tight_layout()
        save_cross_fig(fig, "07_bpe_comparison")

    # ── 5-H. Cross-language Vocabulary Extension Comparison (NEW) ─────────────
    has_vocab_ext = {l: all_dfs[l]["vocab_fragmentation_ratio"].notna().any() for l in langs}
    if any(has_vocab_ext.values()):
        plot_vocab_extension_cross(langs, all_dfs, colors)

    print("[Cross-Language] ✓ All comparison plots done.")


# ═════════════════════════════════════════════════════════════════════════════
# 5-H. VOCABULARY EXTENSION — CROSS-LANGUAGE COMPARISON (NEW)
# ═════════════════════════════════════════════════════════════════════════════

def plot_vocab_extension_cross(
    langs: list[str],
    all_dfs: dict[str, pd.DataFrame],
    colors: dict,
):
    """
    Cross-language comparison of the 3 vocabulary extension metrics.

    Layout (2 rows × 3 cols):
    ┌──────────────────────┬──────────────────────┬────────────────────────┐
    │ Box: Frag Ratio      │ Box: Long-Split Ratio │ Box: Coverage          │
    ├──────────────────────┼──────────────────────┼────────────────────────┤
    │ Bar: mean frag ratio │ Bar: mean LS ratio   │ Verdict heatmap table  │
    │   + threshold line   │   + threshold line   │  (all 3 × all langs)   │
    └──────────────────────┴──────────────────────┴────────────────────────┘
    """
    ext_langs = [l for l in langs if all_dfs[l]["vocab_fragmentation_ratio"].notna().any()]

    fig = plt.figure(figsize=(18, 11))
    fig.patch.set_facecolor(BG_COLOR)
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.35)
    fig.suptitle(
        "Cross-Language Vocabulary Extension Assessment\n"
        "(XLM-RoBERTa xlm-roberta-base — 3 diagnostic metrics)",
        fontsize=14, fontweight="bold", y=1.01,
    )

    metric_cfg = [
        ("vocab_fragmentation_ratio", "① Fragmentation Ratio",     _THRESH_FRAG, True,  0),
        ("vocab_long_split_ratio",    "② Long-Split Ratio (>3)",   _THRESH_LS,   True,  1),
        ("vocab_tokenizer_coverage",  "③ Tokenizer Coverage",      _THRESH_COV,  False, 2),
    ]

    # ── Row 0: box plots per metric ───────────────────────────────────────────
    for col, title, thresh, above_bad, col_idx in metric_cfg:
        ax = fig.add_subplot(gs[0, col_idx])
        data_lists = [all_dfs[l][col].dropna().values for l in ext_langs]
        bp = ax.boxplot(data_lists, patch_artist=True, notch=False,
                        labels=ext_langs, showfliers=True)
        for patch, lang in zip(bp["boxes"], ext_langs):
            patch.set_facecolor(colors[lang])
            patch.set_alpha(0.75)
        for med in bp["medians"]:
            med.set_color("red"); med.set_linewidth(2)

        ax.axhline(thresh, color="#E74C3C", linestyle="--", linewidth=1.8,
                   label=f"Threshold = {thresh}")
        if above_bad:
            ax.axhspan(thresh, ax.get_ylim()[1] if ax.get_ylim()[1] > thresh else thresh * 2,
                       alpha=0.06, color="#E74C3C")
        else:
            ax.axhspan(0, thresh, alpha=0.06, color="#E74C3C")

        ax.set_title(title, fontweight="bold")
        ax.set_ylabel("Value per Document")
        ax.tick_params(axis="x", rotation=15)
        ax.legend(fontsize=8)

    # ── Row 1, col 0: Mean Fragmentation Ratio bar ────────────────────────────
    ax10 = fig.add_subplot(gs[1, 0])
    means_frag = [all_dfs[l]["vocab_fragmentation_ratio"].mean() for l in ext_langs]
    bar_colors = ["#E74C3C" if v > _THRESH_FRAG else "#27AE60" for v in means_frag]
    bars = ax10.bar(ext_langs, means_frag, color=bar_colors, edgecolor="white", width=0.5)
    ax10.axhline(_THRESH_FRAG, color="#E74C3C", linestyle="--", linewidth=1.8,
                 label=f"Threshold = {_THRESH_FRAG}")
    for bar, v in zip(bars, means_frag):
        ax10.text(bar.get_x() + bar.get_width()/2,
                  bar.get_height() + _THRESH_FRAG * 0.02,
                  f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax10.set_title("Mean Fragmentation Ratio\n(BPE tokens ÷ whitespace words)", fontweight="bold")
    ax10.set_ylabel("Mean Value")
    ax10.tick_params(axis="x", rotation=15)
    ax10.legend(fontsize=8)

    # ── Row 1, col 1: Mean Long-Split + Coverage grouped bar ─────────────────
    ax11 = fig.add_subplot(gs[1, 1])
    means_ls  = [all_dfs[l]["vocab_long_split_ratio"].mean()   for l in ext_langs]
    means_cov = [all_dfs[l]["vocab_tokenizer_coverage"].mean() for l in ext_langs]
    x     = np.arange(len(ext_langs))
    width = 0.35

    bc_ls  = ["#E74C3C" if v > _THRESH_LS  else "#27AE60" for v in means_ls]
    bc_cov = ["#E74C3C" if v < _THRESH_COV else "#27AE60" for v in means_cov]

    b1 = ax11.bar(x - width/2, means_ls,  width, color=bc_ls,  edgecolor="white",
                  label="Long-Split Ratio")
    b2 = ax11.bar(x + width/2, means_cov, width, color=bc_cov, edgecolor="white",
                  hatch="//", label="Tokenizer Coverage")
    ax11.axhline(_THRESH_LS,  color="#E74C3C", linestyle="--", linewidth=1.2,
                 label=f"LS threshold={_THRESH_LS}")
    ax11.axhline(_THRESH_COV, color="#F39C12", linestyle=":",  linewidth=1.2,
                 label=f"Cov threshold={_THRESH_COV}")
    for bars in [b1, b2]:
        for bar in bars:
            ax11.text(bar.get_x() + bar.get_width()/2,
                      bar.get_height() + 0.01,
                      f"{bar.get_height():.3f}",
                      ha="center", va="bottom", fontsize=8)
    ax11.set_xticks(x)
    ax11.set_xticklabels(ext_langs, rotation=15)
    ax11.set_title("Mean Long-Split Ratio vs Tokenizer Coverage", fontweight="bold")
    ax11.set_ylabel("Mean Value")
    ax11.legend(fontsize=7)

    # ── Row 1, col 2: Verdict heatmap table ───────────────────────────────────
    ax12 = fig.add_subplot(gs[1, 2])
    ax12.axis("off")

    # Build verdict matrix
    verdict_matrix = []
    col_labels = ["Language", "Frag Ratio", "Long-Split", "Coverage", "Verdict"]
    for lang in ext_langs:
        df = all_dfs[lang]
        mf  = df["vocab_fragmentation_ratio"].mean()
        mls = df["vocab_long_split_ratio"].mean()
        mc  = df["vocab_tokenizer_coverage"].mean()

        fail_frag = mf  > _THRESH_FRAG
        fail_ls   = mls > _THRESH_LS
        fail_cov  = mc  < _THRESH_COV
        n_fail    = sum([fail_frag, fail_ls, fail_cov])

        verdict = (
            "EXTEND" if n_fail >= 2
            else "CONSIDER" if n_fail == 1
            else "OK"
        )
        verdict_matrix.append([
            lang,
            f"{mf:.3f} {'⚠' if fail_frag else '✓'}",
            f"{mls:.3f} {'⚠' if fail_ls  else '✓'}",
            f"{mc:.3f} {'⚠' if fail_cov  else '✓'}",
            verdict,
        ])

    tbl = ax12.table(
        cellText=verdict_matrix,
        colLabels=col_labels,
        cellLoc="center", loc="center",
        bbox=[0.0, 0.05, 1.0, 0.95],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)

    verdict_colors = {"EXTEND": "#FADBD8", "CONSIDER": "#FDEBD0", "OK": "#D5F5E3"}
    verdict_text_colors = {"EXTEND": "#C0392B", "CONSIDER": "#D35400", "OK": "#1E8449"}

    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(GRID_COLOR)
        if r == 0:
            cell.set_facecolor("#2C3E50")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            v_text = verdict_matrix[r - 1][4]
            if c == 4:  # Verdict column
                cell.set_facecolor(verdict_colors.get(v_text, "white"))
                cell.set_text_props(color=verdict_text_colors.get(v_text, "black"),
                                    fontweight="bold")
            elif r % 2 == 0:
                cell.set_facecolor("#F2F3F4")
            else:
                cell.set_facecolor("white")

    ax12.set_title(
        "Vocabulary Extension Verdict\n(⚠ = exceeds threshold  ✓ = within threshold)",
        fontweight="bold", pad=10
    )

    save_cross_fig(fig, "08_vocab_extension_comparison")


# ═════════════════════════════════════════════════════════════════════════════
# 7. SUMMARY STATISTICS REPORT
# ═════════════════════════════════════════════════════════════════════════════

def print_summary(lang: str, df: pd.DataFrame):
    all_words = [w for ws in df["_words"] for w in ws]
    vocab     = set(all_words)
    script    = df["script"].mode()[0] if "script" in df.columns else "latin"
    splitter  = "Khmer terminator ។ / ៕" if script == "khmer" else "Latin .  !  ?  …"
    print(f"\n{'═'*60}")
    print(f"  CORPUS SUMMARY — {lang}")
    print(f"{'═'*60}")
    print(f"  Script detected    : {script.upper()}  →  splitter: {splitter}")
    print(f"  Documents          : {len(df):,}")
    print(f"  Total tokens       : {len(all_words):,}")
    print(f"  Vocabulary size    : {len(vocab):,}")
    print(f"  Avg words/doc      : {df['word_count'].mean():.1f}  (±{df['word_count'].std():.1f})")
    print(f"  Total sentences    : {df['sent_count'].sum():,}")
    print(f"  Avg sentences/doc  : {df['sent_count'].mean():.1f}  (±{df['sent_count'].std():.1f})")
    print(f"  Avg TTR            : {df['type_token_ratio'].mean():.3f}")
    print(f"  Avg word length    : {df['avg_word_len'].mean():.2f} chars")
    if df["bpe_token_count"].notna().any():
        print(f"  ── XLM-RoBERTa BPE (xlm-roberta-base) ──")
        print(f"  Total BPE tokens   : {int(df['bpe_token_count'].sum()):,}")
        print(f"  Avg BPE/doc        : {df['bpe_token_count'].mean():.1f}  (±{df['bpe_token_count'].std():.1f})")
        print(f"  Avg fertility      : {df['bpe_fertility'].mean():.3f}")
    if df["vocab_fragmentation_ratio"].notna().any():
        print(f"  ── Vocabulary Extension Metrics ──")
        print(f"  Fragmentation ratio: {df['vocab_fragmentation_ratio'].mean():.3f}  "
              f"(threshold > {_THRESH_FRAG})")
        print(f"  Long-split ratio   : {df['vocab_long_split_ratio'].mean():.3f}  "
              f"(threshold > {_THRESH_LS})")
        print(f"  Tokenizer coverage : {df['vocab_tokenizer_coverage'].mean():.3f}  "
              f"(threshold < {_THRESH_COV})")
    print(f"  Categories         : {df['category'].nunique()} ({', '.join(df['category'].unique()[:5])})")
    top5 = Counter(all_words).most_common(5)
    print(f"  Top-5 words        : {', '.join(f'{w}({c})' for w,c in top5)}")
    print(f"{'─'*60}")


# ═════════════════════════════════════════════════════════════════════════════
# 8. MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="EDA for Low-Resource Language Corpus")
    parser.add_argument("--data_dir",  default="processed_data",
                        help="Directory containing .json corpus files")
    parser.add_argument("--langs",     nargs="*", default=None,
                        help="Specific language labels (must match filenames without .json)")
    parser.add_argument("--single",    action="store_true",
                        help="Run single-language plots only (no cross-language comparison)")
    parser.add_argument("--no-bpe",    action="store_true",
                        help="Skip XLM-RoBERTa BPE tokenisation (faster, offline)")
    parser.add_argument("--bpe-batch", type=int, default=64,
                        help="Batch size for BPE tokenisation (default: 64)")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("  Low-Resource Language Corpus — EDA")
    print(f"  Data dir : {args.data_dir}")
    print(f"  Output   : {OUTPUT_DIR}/")
    print(f"  BPE      : {'disabled (--no-bpe)' if args.no_bpe else 'xlm-roberta-base'}")
    print(f"{'='*60}\n")

    if args.langs:
        lang_files = {l.title(): f"{l.lower()}.json" for l in args.langs}
    else:
        lang_files = None

    # ── Step 1: Load corpora ──────────────────────────────────────────────────
    print("[1/4] Loading corpora …")
    corpora = load_multiple_corpora(args.data_dir, lang_files)

    if not corpora:
        print("  ERROR: No corpus files found. Check --data_dir.")
        return

    # ── Step 2: BPE tokenisation + feature extraction ─────────────────────────
    print("\n[2/4] Tokenising & extracting features …")
    all_dfs:   dict[str, pd.DataFrame] = {}
    all_stats: dict[str, dict]         = {}

    for lang, records in corpora.items():
        if not args.no_bpe:
            bpe_results = bpe_tokenize_corpus(records, lang, batch_sz=args.bpe_batch)
        else:
            bpe_results = None

        df = extract_features(records, bpe_results=bpe_results)
        all_dfs[lang] = df
        print_summary(lang, df)

        # Build & save per-language JSON stats
        stats = build_lang_stats(lang, df)
        all_stats[lang] = stats
        save_lang_json(lang, stats)

    # ── Step 3: Visualisations ────────────────────────────────────────────────
    print("\n[3/4] Generating visualizations …")
    for lang, df in all_dfs.items():
        plot_single_language(lang, df)

    if not args.single:
        plot_cross_language(all_dfs)

    # ── Step 4: Cross-language JSON ───────────────────────────────────────────
    print("\n[4/4] Saving cross-language JSON stats …")
    if not args.single:
        save_cross_language_json(all_stats)

    print(f"\nEDA complete!\n"
          f"   Charts & JSON stats saved to: {OUTPUT_DIR}/\n"
          f"   Structure:\n"
          f"     {OUTPUT_DIR}/\n"
          f"       cham/          ← plots + stats.json\n"
          f"       khmer/         ← plots + stats.json\n"
          f"       tay-nung/      ← plots + stats.json\n"
          f"       cross_language/ ← plots + stats.json (all 3 languages)\n")


if __name__ == "__main__":
    main()