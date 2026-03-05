"""
cham_khmer_parser.py
====================
Parse raw text files from the data/ directory structure into cham.json and khmer.json.

Directory structure expected:
    data/
    ├── Chăm/VOV/
    │   ├── BAO_TON_VAN_HOA_CHAM/
    │   │   ├── file1.txt
    │   │   └── ...
    │   └── ...
    └── Khmer/
        ├── AnGiang_gov/
        │   ├── file1.txt
        │   └── ...
        └── ...

Each raw text file format:
    URL: <url>
    TIÊU ĐỀ: <title>
    TÓM TẮT: <summary>
    TAGS: <tags>

    --- NỘI DUNG ---

    <content>
"""

import os
import json
import re
import sys
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    print("[INFO] tqdm not found. Install with: pip install tqdm")
    print("[INFO] Falling back to basic progress output.\n")
    def tqdm(iterable, desc="", unit=""):
        total = len(iterable) if hasattr(iterable, "__len__") else "?"
        print(f"→ {desc}: processing {total} {unit}(s)...")
        for i, item in enumerate(iterable, 1):
            if isinstance(total, int) and (i % max(1, total // 10) == 0 or i == total):
                print(f"   {i}/{total} done")
            yield item


# ── Helpers ────────────────────────────────────────────────────────────────────

def parse_file(filepath: Path) -> dict:
    """Parse a single raw text file into a structured dict."""
    try:
        text = filepath.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return None

    record = {
        "url": "",
        "title": "",
        "summary": "",
        "tags": [],
        "content": "",
    }

    # Split header from content at the separator line
    separator = "--- NỘI DUNG ---"
    if separator in text:
        header_part, content_part = text.split(separator, 1)
        record["content"] = content_part.strip()
    else:
        header_part = text
        record["content"] = ""

    # Parse header fields (multiline-aware)
    header_lines = header_part.splitlines()
    current_key = None
    buffer = []

    def flush_buffer(key, buf):
        value = " ".join(buf).strip()
        if key == "url":
            record["url"] = value
        elif key == "title":
            record["title"] = value
        elif key == "summary":
            record["summary"] = value
        elif key == "tags":
            # Split on commas, semicolons, or pipe chars; filter blanks
            raw_tags = re.split(r"[,;|]+", value)
            record["tags"] = [t.strip() for t in raw_tags if t.strip()]

    field_map = {
        r"^URL\s*:": "url",
        r"^TIÊU ĐỀ\s*:": "title",
        r"^TÓM TẮT\s*:": "summary",
        r"^TAGS\s*:": "tags",
    }

    for line in header_lines:
        matched = False
        for pattern, key in field_map.items():
            if re.match(pattern, line, re.IGNORECASE):
                if current_key is not None:
                    flush_buffer(current_key, buffer)
                current_key = key
                # Value is everything after the colon
                after_colon = re.sub(pattern, "", line, flags=re.IGNORECASE).strip()
                buffer = [after_colon] if after_colon else []
                matched = True
                break
        if not matched and current_key is not None:
            # Continuation line
            if line.strip():
                buffer.append(line.strip())

    if current_key is not None:
        flush_buffer(current_key, buffer)

    return record


def collect_files(root: Path) -> list[tuple[str, str, Path]]:
    """
    Walk the root directory and collect (group, site, category, filepath) tuples.

    Returns list of (group_name, site_name, category_name, filepath).
    group_name  → "Cham" or "Khmer"
    site_name   → e.g. "VOV", "AnGiang_gov"
    category    → sub-folder inside the site (may be empty string if files sit directly in site folder)
    """
    entries = []
    for group_dir in sorted(root.iterdir()):
        if not group_dir.is_dir():
            continue
        group_name = group_dir.name  # "Chăm" / "VOV" nesting or "Khmer"

        # Handle the nested "Chăm/VOV" pattern where there may be an extra level
        # Determine the target label
        if "ham" in group_name or "Cham" in group_name.lower() or group_name == "Chăm":
            group_label = "Cham"
        elif "Khmer" in group_name:
            group_label = "Khmer"
        elif "y-N" in group_name or "Tay" in group_name or "Nùng" in group_name:
            group_label = "Tay-Nung"
        else:
            group_label = group_name  # fallback: keep as-is

        # Walk one or two levels to find site folders
        for site_or_cat in sorted(group_dir.iterdir()):
            if not site_or_cat.is_dir():
                # txt files directly under group — treat group as site
                if site_or_cat.suffix in (".txt", ".text"):
                    entries.append((group_label, group_name, "", site_or_cat))
                continue

            # Check if this folder contains txt files directly (= site folder with no category)
            # or sub-folders (= site contains categories)
            sub_items = list(site_or_cat.iterdir())
            has_subdirs = any(s.is_dir() for s in sub_items)
            has_txt = any(s.is_file() and s.suffix in (".txt", ".text") for s in sub_items)

            site_name = site_or_cat.name

            if has_subdirs:
                # site_or_cat is a site; sub-folders are categories
                for cat_dir in sorted(site_or_cat.iterdir()):
                    if cat_dir.is_dir():
                        for f in sorted(cat_dir.rglob("*.txt")):
                            entries.append((group_label, site_name, cat_dir.name, f))
                        for f in sorted(cat_dir.rglob("*.text")):
                            entries.append((group_label, site_name, cat_dir.name, f))
                    elif cat_dir.is_file() and cat_dir.suffix in (".txt", ".text"):
                        entries.append((group_label, site_name, "", cat_dir))
            elif has_txt:
                # Files sit directly in the site folder (no category)
                for f in sorted(site_or_cat.glob("*.txt")):
                    entries.append((group_label, site_name, "", f))
                for f in sorted(site_or_cat.glob("*.text")):
                    entries.append((group_label, site_name, "", f))

    return entries


# ── Main ───────────────────────────────────────────────────────────────────────

def cham_khmer_parser(data_dir: str = "data", output_dir: str = "."):
    root = Path(data_dir)
    if not root.exists():
        print(f"[ERROR] Directory '{data_dir}' does not exist.")
        sys.exit(1)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"📂 Scanning directory: {root.resolve()}")
    entries = collect_files(root)

    if not entries:
        print("[WARNING] No .txt files found. Check your directory structure.")
        return

    print(f"✅ Found {len(entries)} file(s) to process.\n")

    # Separate into groups
    group_buckets: dict[str, list] = {}

    skipped = 0
    for group_label, site_name, category, filepath in tqdm(entries, desc="Parsing files", unit="file"):
        record = parse_file(filepath)
        if record is None:
            skipped += 1
            continue

        record["site"] = site_name
        record["category"] = category
        record["source_file"] = str(filepath)

        # Reorder keys for readability
        ordered = {
            "site": record["site"],
            "category": record["category"],
            "url": record["url"],
            "title": record["title"],
            "summary": record["summary"],
            "tags": record["tags"],
            "content": record["content"],
        }

        group_buckets.setdefault(group_label, []).append(ordered)

    print(f"\n⚠️  Skipped {skipped} file(s) due to read errors.\n")

    # Write output files
    output_map = {
        "Cham": "cham.json",
        "Khmer": "khmer.json",
        "Tay-Nung": "tay_nung.json",
    }

    for group_label, records in group_buckets.items():
        filename = output_map.get(group_label, f"{group_label.lower()}.json")
        out_file = output_path / filename
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"💾 [{group_label}] {len(records)} records → {out_file}")

    print("\n✅ Done!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parse Cham/Khmer corpus text files into JSON.")
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Root directory containing the corpus (default: data/)",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory to write output JSON files (default: current directory)",
    )
    args = parser.parse_args()

    cham_khmer_parser(data_dir=args.data_dir, output_dir=args.output_dir)