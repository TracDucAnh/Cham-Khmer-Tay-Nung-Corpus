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

# Khmer Unicode block: U+1780–U+17FF
_KHMER_RE   = re.compile(r'[\u1780-\u17FF]')

# Khmer sentence terminators: ។ (U+17D4)  ៕ (U+17D5)  ៖ (U+17D6)
_SENT_KHMER = re.compile(r'[។៕]+')

# Latin-script sentence endings (Cham, Tày-Nùng)
# Split on: .  !  ?  …  plus optional surrounding whitespace/newlines
_SENT_LATIN = re.compile(r'(?<=[^\d])[.!?…]+(?=\s|$)|(?<=\n)\n+')


def detect_script(text: str) -> str:
    """
    Return 'khmer' if the text contains Khmer Unicode characters,
    otherwise return 'latin' (covers Cham, Tày-Nùng, Vietnamese, etc.).
    """
    return "khmer" if _KHMER_RE.search(text) else "latin"


def split_sentences(text: str, script: str) -> list[str]:
    """
    Split *text* into sentences using script-appropriate rules.

    Khmer  → split on ។  ៕  (also collapse blank lines)
    Latin  → split on . ! ? … (guard against decimal numbers like 5.3)
             blank-line boundaries also count as sentence boundaries
    """
    if not text:
        return []

    if script == "khmer":
        # Primary split on Khmer terminators
        parts = _SENT_KHMER.split(text)
    else:
        # For Latin scripts: split on sentence-ending punctuation
        # Also treat double-newlines (paragraph breaks) as boundaries
        normalised = re.sub(r'\n{2,}', '. ', text)   # paragraph → "."
        parts = _SENT_LATIN.split(normalised)

    return [p.strip() for p in parts if p.strip()]


# ═════════════════════════════════════════════════════════════════════════════
# 2-B. WORD SEGMENTATION
# ═════════════════════════════════════════════════════════════════════════════

# Khmer has NO whitespace between words → requires a dedicated segmenter.
# We use `khmernltk` (pip install khmernltk) which ships a pre-trained
# CRF/MaxEnt word-boundary model for Khmer.
#
# Latin-script languages (Cham, Tày-Nùng) use whitespace tokenisation with
# light punctuation stripping — appropriate for these languages.

_khmer_word_seg = None   # lazy-loaded once


def _load_khmer_segmenter():
    """Load khmernltk word segmenter, warn gracefully if not installed."""
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
        # Fallback: split on Unicode Khmer character clusters
        _khmer_word_seg = _khmer_cluster_fallback
    return _khmer_word_seg


def _khmer_cluster_fallback(text: str) -> list[str]:
    """
    Fallback Khmer tokeniser when khmernltk is unavailable.
    Splits on whitespace and Khmer punctuation; NOT linguistically accurate
    but avoids single-character counts.
    """
    # Remove punctuation, split on whitespace/punctuation boundaries
    tokens = re.findall(r'[\u1780-\u17FF]+', text)
    return [t for t in tokens if t.strip()]


def tokenize(text: str, script: str = "latin") -> list[str]:
    """
    Script-aware word tokeniser.

    - latin  → regex whitespace tokeniser (Cham, Tày-Nùng, Vietnamese)
    - khmer  → khmernltk word segmenter (CRF-based, handles no-space script)

    Returns a list of lowercase token strings.
    """
    if not text:
        return []

    if script == "khmer":
        seg = _load_khmer_segmenter()
        raw_tokens = seg(text)
        # khmernltk may return tokens with spaces or punctuation — clean up
        return [
            t.strip().lower()
            for t in raw_tokens
            if t.strip() and not re.fullmatch(r'[\s\u200b\u00a0]+', t)
               and re.search(r'[\u1780-\u17FF]', t)   # keep only Khmer-char tokens
        ]
    else:
        # Latin: standard unicode word boundary split
        return re.findall(r'\b\w+\b', text.lower(), flags=re.UNICODE)



# ═════════════════════════════════════════════════════════════════════════════
# 3. XLM-RoBERTa BPE TOKENIZER
# ═════════════════════════════════════════════════════════════════════════════

_XLM_MODEL_NAME = "xlm-roberta-base"
_bpe_tokenizer  = None   # loaded lazily once


def get_bpe_tokenizer() -> AutoTokenizer:
    """Load XLM-RoBERTa tokenizer (cached after first call)."""
    global _bpe_tokenizer
    if _bpe_tokenizer is None:
        print(f"  Loading XLM-RoBERTa tokenizer ({_XLM_MODEL_NAME}) …")
        _bpe_tokenizer = AutoTokenizer.from_pretrained(
            _XLM_MODEL_NAME,
            use_fast=True,        # Rust-backed HuggingFace fast tokenizer
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

    Returns a list of dicts:
        {
          "bpe_token_count":    int,   # total subword tokens (excl. special tokens)
          "bpe_unique_tokens":  int,   # unique subword IDs
          "bpe_tokens":         list[str],  # decoded subword strings (for freq analysis)
        }

    Progress is shown with tqdm.
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
        # encode_batch without truncation so we count ALL tokens
        encoded = tok(
            batch,
            add_special_tokens=False,   # exclude <s> and </s>
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

        # Per-sentence word counts using the same script-aware tokeniser
        sent_word_counts = [len(tokenize(s, script=script)) for s in sentences]

        row = {
            "site":             site,
            "category":         cat,
            "n_tags":           len(tags),
            "title_len":        len(title.split()),
            "script":           script,
            # content stats
            "char_count":       len(full_text),
            "word_count":       len(words),
            "unique_words":     len(set(words)),
            "sent_count":       len(sentences),
            "para_count":       len(paragraphs),
            "type_token_ratio": len(set(words)) / max(len(words), 1),
            "avg_word_len":     np.mean([len(w) for w in words]) if words else 0,
            "avg_sent_len":     np.mean(sent_word_counts) if sent_word_counts else 0,
            # raw text (for word freq)
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
                                               / max(len(words), 1))   # BPE/whitespace ratio
            row["bpe_type_token_ratio"]     = (bpe["bpe_unique_tokens"]
                                               / max(bpe["bpe_token_count"], 1))
            row["_bpe_tokens"]              = bpe["bpe_tokens"]
        else:
            row["bpe_token_count"]          = np.nan
            row["bpe_unique_tokens"]        = np.nan
            row["bpe_fertility"]            = np.nan
            row["bpe_type_token_ratio"]     = np.nan
            row["_bpe_tokens"]              = []

        rows.append(row)
    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════════
# 4. PLOT HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def save_fig(fig, name: str):
    out = OUTPUT_DIR / f"{name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def stat_box(ax, df: pd.DataFrame, col: str):
    """Annotate an axes with basic statistics."""
    mu  = df[col].mean()
    med = df[col].median()
    ax.axvline(mu,  color="red",    linestyle="--", linewidth=1.2, label=f"Mean={mu:.1f}")
    ax.axvline(med, color="orange", linestyle=":",  linewidth=1.2, label=f"Median={med:.1f}")
    ax.legend()


# ═════════════════════════════════════════════════════════════════════════════
# 4. SINGLE-LANGUAGE EDA
# ═════════════════════════════════════════════════════════════════════════════

def plot_single_language(lang: str, df: pd.DataFrame):
    print(f"\n[{lang}] Generating single-language plots …")

    # ── 4-A. Summary statistics card ─────────────────────────────────────────
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
    save_fig(fig, f"{lang.lower()}_01_univariate")

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
        save_fig(fig, f"{lang.lower()}_02_correlation")

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
        save_fig(fig, f"{lang.lower()}_03_category_dist")

    # ── 4-D. Word count vs TTR scatter ────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{lang} — Vocabulary Richness", fontsize=13, fontweight="bold")

    axes[0].scatter(df["word_count"], df["type_token_ratio"],
                    alpha=0.6, color=PALETTE[1], edgecolors="white", s=40)
    axes[0].set_xlabel("Word Count")
    axes[0].set_ylabel("Type-Token Ratio (TTR)")
    axes[0].set_title("Word Count vs. TTR\n(higher TTR = richer vocabulary)")

    axes[1].scatter(df["word_count"], df["unique_words"],
                    alpha=0.6, color=PALETTE[2], edgecolors="white", s=40)
    # Heaps' law reference line
    x_vals = np.linspace(df["word_count"].min(), df["word_count"].max(), 200)
    axes[1].plot(x_vals, x_vals**0.67 * (df["unique_words"].max() / df["word_count"].max()**0.67),
                 "r--", linewidth=1, label="Heaps' law ref (~N^0.67)")
    axes[1].set_xlabel("Word Count (Tokens)")
    axes[1].set_ylabel("Unique Words (Types)")
    axes[1].set_title("Token–Type Growth")
    axes[1].legend()

    fig.tight_layout()
    save_fig(fig, f"{lang.lower()}_04_vocabulary_richness")

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
    save_fig(fig, f"{lang.lower()}_05_top_words")

    # ── 4-F. Zipf's law log-log ───────────────────────────────────────────────
    ranks  = np.arange(1, len(freq) + 1)
    counts = np.array([c for _, c in freq.most_common()])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.loglog(ranks, counts, "o", markersize=2, alpha=0.5, color=PALETTE[4], label="Observed")
    # Zipf reference: freq ∝ 1/rank
    zipf_ref = counts[0] / ranks
    ax.loglog(ranks, zipf_ref, "r--", linewidth=1.5, label="Zipf (α=1) reference")
    ax.set_xlabel("Rank (log scale)")
    ax.set_ylabel("Frequency (log scale)")
    ax.set_title(f"{lang} — Zipf's Law Analysis")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, f"{lang.lower()}_06_zipf_law")

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
        save_fig(fig, f"{lang.lower()}_07_wordcloud")
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
        # annotate stats
        q1, med, q3 = df[col].quantile([0.25, 0.5, 0.75])
        ax.text(1.35, med, f"Med={med:.0f}", va="center", fontsize=8, color="red")
        ax.text(1.35, q3,  f"Q3={q3:.0f}",  va="center", fontsize=8)
        ax.text(1.35, q1,  f"Q1={q1:.0f}",  va="center", fontsize=8)

    fig.tight_layout()
    save_fig(fig, f"{lang.lower()}_08_boxplots")

    # ── 4-I. Sentence count per document (language-aware splitter) ────────────
    script_label = df["script"].mode()[0].upper() if "script" in df.columns else "LATIN"
    sent_counts  = df["sent_count"].dropna().astype(int)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle(
        f"{lang} — Sentence Count per Document\n"
        f"(Script: {script_label} | "
        f"{'Khmer terminator ។ / ៕' if script_label == 'KHMER' else 'Latin terminators . ! ? …'})",
        fontsize=13, fontweight="bold"
    )

    # Left: histogram
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

    # Middle: sorted bar chart (each doc as one bar — capped at 200 docs for readability)
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

    # Right: stats summary table
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
    save_fig(fig, f"{lang.lower()}_09_sentence_counts")

    # ── 4-J. BPE Token Analysis (XLM-RoBERTa) ────────────────────────────────
    if df["bpe_token_count"].notna().any():
        bpe_counts  = df["bpe_token_count"].dropna()
        fertility   = df["bpe_fertility"].dropna()       # BPE tokens / whitespace words
        bpe_ttr     = df["bpe_type_token_ratio"].dropna()

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(
            f"{lang} — XLM-RoBERTa BPE Token Analysis\n"
            f"(model: xlm-roberta-base  |  special tokens excluded)",
            fontsize=13, fontweight="bold"
        )

        # (0,0) Histogram: BPE token count per doc
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

        # (0,1) Scatter: whitespace words vs BPE tokens (fertility cloud)
        ax = axes[0, 1]
        ax.scatter(df["word_count"].dropna(), bpe_counts,
                   alpha=0.5, color=PALETTE[1], edgecolors="none", s=25)
        # ideal 1:1 line
        lim = max(df["word_count"].max(), bpe_counts.max())
        ax.plot([0, lim], [0, lim], "k--", lw=0.8, label="1:1 (no fragmentation)")
        ax.set_title("Whitespace Words vs BPE Tokens\n(points above line = more fragmentation)")
        ax.set_xlabel("Whitespace Word Count")
        ax.set_ylabel("BPE Token Count")
        ax.legend(fontsize=8)

        # (0,2) Fertility histogram
        ax = axes[0, 2]
        ax.hist(fertility, bins=min(40, max(5, len(fertility)//3)),
                color=PALETTE[2], edgecolor="white")
        ax.axvline(fertility.mean(),   color="red",    linestyle="--", lw=1.4,
                   label=f"Mean={fertility.mean():.2f}")
        ax.axvline(1.0,                color="black",  linestyle=":",  lw=1.0,
                   label="Fertility=1 (no split)")
        ax.set_title("BPE Fertility per Document\n(BPE tokens ÷ whitespace words)")
        ax.set_xlabel("Fertility")
        ax.set_ylabel("Frequency")
        ax.legend()

        # (1,0) Top-30 BPE subwords
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

        # (1,1) BPE TTR vs whitespace TTR scatter
        ax = axes[1, 1]
        ax.scatter(df["type_token_ratio"].dropna(), bpe_ttr,
                   alpha=0.5, color=PALETTE[4], edgecolors="none", s=25)
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Equal TTR")
        ax.set_title("Whitespace TTR vs BPE TTR\n(vocab richness comparison)")
        ax.set_xlabel("Whitespace Type-Token Ratio")
        ax.set_ylabel("BPE Type-Token Ratio")
        ax.legend(fontsize=8)

        # (1,2) Summary stats table
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
        save_fig(fig, f"{lang.lower()}_10_bpe_analysis")

    print(f"[{lang}] ✓ All single-language plots done.")


# ═════════════════════════════════════════════════════════════════════════════
# 5. CROSS-LANGUAGE COMPARISON
# ═════════════════════════════════════════════════════════════════════════════

def plot_cross_language(all_dfs: dict[str, pd.DataFrame]):
    if len(all_dfs) < 2:
        print("  Skipping cross-language plots (only 1 language loaded).")
        return

    print("\n[Cross-Language] Generating comparison plots …")

    langs  = list(all_dfs.keys())
    colors = {lang: PALETTE[i] for i, lang in enumerate(langs)}

    # ── 5-A. Side-by-side box plots for key metrics ───────────────────────────
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
    save_fig(fig, "cross_01_comparison_boxplots")

    # ── 5-B. Corpus size summary bar chart ────────────────────────────────────
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
    save_fig(fig, "cross_02_corpus_size")

    # ── 5-C. Vocabulary overlap (Jaccard similarity) ──────────────────────────
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
    ax.set_title("Vocabulary Overlap — Jaccard Similarity\n(0=no overlap, 1=identical)")
    fig.tight_layout()
    save_fig(fig, "cross_03_vocab_jaccard")

    # ── 5-D. KDE distribution overlay for word count ─────────────────────────
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
    save_fig(fig, "cross_04_kde_overlay")

    # ── 5-E. Top-N word frequency heatmap across languages ────────────────────
    top_per_lang = {}
    for lang, df in all_dfs.items():
        all_words = [w for ws in df["_words"] for w in ws]
        top_per_lang[lang] = dict(Counter(all_words).most_common(50))

    all_top_words = list({w for d in top_per_lang.values() for w in list(d.keys())[:20]})
    heat_data = pd.DataFrame(
        {lang: [top_per_lang[lang].get(w, 0) for w in all_top_words] for lang in langs},
        index=all_top_words
    )
    heat_norm = heat_data.div(heat_data.max()).fillna(0)  # normalize 0–1

    fig, ax = plt.subplots(figsize=(max(8, len(langs)*2), max(10, len(all_top_words)*0.4)))
    sns.heatmap(heat_norm, cmap="Blues", linewidths=0.3, ax=ax,
                cbar_kws={"label": "Normalized Frequency"})
    ax.set_title("Top Word Frequency Heatmap Across Languages\n(normalized per language)")
    ax.set_xlabel("Language")
    ax.set_ylabel("Word")
    fig.tight_layout()
    save_fig(fig, "cross_05_word_freq_heatmap")

    # ── 5-F. Cross-language sentence count comparison ─────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Cross-Language Sentence Count Comparison\n"
                 "(Cham/Tày-Nùng → Latin splitter .!?…  |  Khmer → ។ / ៕ splitter)",
                 fontsize=13, fontweight="bold")

    # Left: box plot per language
    ax = axes[0]
    data_lists = [all_dfs[l]["sent_count"].dropna().values for l in langs]
    bp = ax.boxplot(data_lists, patch_artist=True, notch=False,
                    labels=langs, showfliers=True)
    for patch, lang in zip(bp["boxes"], langs):
        patch.set_facecolor(colors[lang])
        patch.set_alpha(0.7)
    for median in bp["medians"]:
        median.set_color("red")
        median.set_linewidth(2)
    ax.set_title("Sentence Count Distribution\nper Language")
    ax.set_ylabel("Sentence Count per Document")
    ax.tick_params(axis="x", rotation=15)

    # Middle: mean ± std bar chart
    ax = axes[1]
    means  = [all_dfs[l]["sent_count"].mean()  for l in langs]
    stds   = [all_dfs[l]["sent_count"].std()   for l in langs]
    bars   = ax.bar(langs, means, yerr=stds, color=[colors[l] for l in langs],
                    edgecolor="white", capsize=6, error_kw={"linewidth": 1.5})
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + s + max(means)*0.01,
                f"{m:.1f}±{s:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_title("Mean Sentence Count per Doc\n(± 1 std)")
    ax.set_ylabel("Mean Sentence Count")
    ax.tick_params(axis="x", rotation=15)

    # Right: per-language KDE overlay
    ax = axes[2]
    for lang in langs:
        data = all_dfs[lang]["sent_count"].dropna()
        script = all_dfs[lang]["script"].mode()[0] if "script" in all_dfs[lang].columns else "latin"
        splitter_note = "Khmer ។/៕" if script == "khmer" else "Latin .!?…"
        if len(data) > 1:
            sns.kdeplot(data, ax=ax, label=f"{lang} ({splitter_note})",
                        color=colors[lang], linewidth=2, fill=True, alpha=0.15)
        else:
            ax.axvline(data.iloc[0], color=colors[lang], linewidth=2,
                       label=f"{lang} (n=1)")
    ax.set_xlabel("Sentence Count per Document")
    ax.set_ylabel("Density")
    ax.set_title("KDE of Sentence Counts\n(per language & splitter)")
    ax.legend()

    fig.tight_layout()
    save_fig(fig, "cross_06_sentence_count_comparison")

    # ── 5-G. Cross-language BPE token comparison ──────────────────────────────
    has_bpe = {l: all_dfs[l]["bpe_token_count"].notna().any() for l in langs}
    if any(has_bpe.values()):
        bpe_langs = [l for l in langs if has_bpe[l]]

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(
            "Cross-Language BPE Token Analysis — XLM-RoBERTa (xlm-roberta-base)\n"
            "Special tokens excluded  |  Fertility = BPE tokens ÷ whitespace words",
            fontsize=13, fontweight="bold"
        )

        # Left: BPE token count box plot
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

        # Middle: Fertility box plot
        ax = axes[1]
        data_fert = [all_dfs[l]["bpe_fertility"].dropna().values for l in bpe_langs]
        bp2 = ax.boxplot(data_fert, patch_artist=True, labels=bpe_langs, showfliers=True)
        for patch, l in zip(bp2["boxes"], bpe_langs):
            patch.set_facecolor(colors[l]); patch.set_alpha(0.7)
        for med in bp2["medians"]:
            med.set_color("red"); med.set_linewidth(2)
        ax.axhline(1.0, color="black", linestyle="--", lw=1.0, label="Fertility=1")
        ax.set_title("BPE Fertility per Language\n(higher = more subword fragmentation)")
        ax.set_ylabel("Fertility (BPE / whitespace)")
        ax.legend(fontsize=8)
        ax.tick_params(axis="x", rotation=15)

        # Right: mean BPE vs mean whitespace bar comparison
        ax = axes[2]
        x      = np.arange(len(bpe_langs))
        width  = 0.35
        means_ws  = [all_dfs[l]["word_count"].mean()    for l in bpe_langs]
        means_bpe = [all_dfs[l]["bpe_token_count"].mean() for l in bpe_langs]
        b1 = ax.bar(x - width/2, means_ws,  width, label="Whitespace words",
                    color=[colors[l] for l in bpe_langs], alpha=0.6, edgecolor="white")
        b2 = ax.bar(x + width/2, means_bpe, width, label="BPE tokens",
                    color=[colors[l] for l in bpe_langs], alpha=1.0, edgecolor="white",
                    hatch="//")
        for bars in [b1, b2]:
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + max(means_bpe)*0.01,
                        f"{bar.get_height():.0f}",
                        ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(bpe_langs, rotation=15)
        ax.set_title("Mean Whitespace Words vs BPE Tokens\nper Document")
        ax.set_ylabel("Mean Count per Document")
        ax.legend()

        fig.tight_layout()
        save_fig(fig, "cross_07_bpe_comparison")

    print("[Cross-Language] ✓ All comparison plots done.")


# ═════════════════════════════════════════════════════════════════════════════
# 6. SUMMARY STATISTICS REPORT
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
        print(f"  Avg fertility      : {df['bpe_fertility'].mean():.3f}  "
              f"(BPE tokens per whitespace word)")
    print(f"  Categories         : {df['category'].nunique()} ({', '.join(df['category'].unique()[:5])})")
    top5 = Counter(all_words).most_common(5)
    print(f"  Top-5 words        : {', '.join(f'{w}({c})' for w,c in top5)}")
    print(f"{'─'*60}")


# ═════════════════════════════════════════════════════════════════════════════
# 7. MAIN
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

    # Build lang→file mapping
    if args.langs:
        lang_files = {l.title(): f"{l.lower()}.json" for l in args.langs}
    else:
        lang_files = None   # auto-detect

    # ── Step 1: Load corpora ──────────────────────────────────────────────────
    print("[1/3] Loading corpora …")
    corpora = load_multiple_corpora(args.data_dir, lang_files)

    if not corpora:
        print("  ERROR: No corpus files found. Check --data_dir.")
        return

    # ── Step 2: BPE tokenisation + feature extraction ─────────────────────────
    print("\n[2/3] Tokenising & extracting features …")
    all_dfs: dict[str, pd.DataFrame] = {}

    for lang, records in corpora.items():
        # Run XLM-RoBERTa BPE tokeniser with tqdm progress bar
        if not args.no_bpe:
            bpe_results = bpe_tokenize_corpus(records, lang, batch_sz=args.bpe_batch)
        else:
            bpe_results = None

        df = extract_features(records, bpe_results=bpe_results)
        all_dfs[lang] = df
        print_summary(lang, df)

    # ── Step 3: Visualisations ────────────────────────────────────────────────
    print("\n[3/3] Generating visualizations …")
    for lang, df in all_dfs.items():
        plot_single_language(lang, df)

    if not args.single:
        plot_cross_language(all_dfs)

    print(f"\n✅  EDA complete! All charts saved to: {OUTPUT_DIR}/\n")


if __name__ == "__main__":
    main()