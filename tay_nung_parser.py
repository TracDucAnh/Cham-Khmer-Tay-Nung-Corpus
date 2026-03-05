"""
tay_nung_parser.py
==================
Parse JSON files from the Tày-Nùng corpus directory into a single tay_nung.json.

Expected directory structure:
    data/
    └── Tày-Nùng/
        └── VOV/                        ← site name
            ├── am_nhac_tay_nung.json   ← category name (filename without .json)
            ├── dan_ca.json
            └── ...

Each source JSON file is a list of records with at least:
    {
        "title":   "...",
        "content": "...",
        "url":     "..."
    }

Output record fields:
    site, category, url, title, summary, content
"""

import json
import re
import sys
from pathlib import Path

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, desc="", unit=""):
        items = list(iterable)
        total = len(items)
        print(f"→ {desc}: {total} {unit}(s)")
        for i, item in enumerate(items, 1):
            if total <= 10 or i % max(1, total // 10) == 0 or i == total:
                print(f"   {i}/{total}", end="\r")
            yield item
        print()


# ── Helpers ────────────────────────────────────────────────────────────────────

def extract_summary(content: str, max_chars: int = 200) -> str:
    """
    Try to extract a summary from the content field.
    Heuristic: take the first meaningful paragraph/line, up to max_chars.
    Strips common noise like tag lists, author lines, etc.
    """
    if not content:
        return ""

    # Split into non-empty lines
    lines = [l.strip() for l in content.splitlines() if l.strip()]

    # Skip lines that look like metadata / noise
    noise_patterns = [
        r"^Tags\s*[:：]?$",
        r"^Viết bình luận",
        r"^Tên của bạn",
        r"^Thư điện tử",
        r"^Mật khẩu",
        r"^\d{1,2}:\d{2},?\s+\d{1,2}/\d{1,2}/\d{4}$",   # timestamp line
        r"^(Thứ \w+|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)",
    ]

    # Find the index where Tags block starts — cut everything from there
    tag_start = None
    for i, line in enumerate(lines):
        if re.match(r"^Tags\s*[:：]?$", line, re.IGNORECASE):
            tag_start = i
            break
    if tag_start is not None:
        lines = lines[:tag_start]

    candidate_lines = []
    for line in lines:
        skip = any(re.search(p, line, re.IGNORECASE) for p in noise_patterns)
        if not skip:
            candidate_lines.append(line)

    # Skip first line (usually repeats the title)
    summary_lines = candidate_lines[1:] if candidate_lines else []

    summary = " ".join(summary_lines)
    if len(summary) > max_chars:
        summary = summary[:max_chars].rsplit(" ", 1)[0] + "…"
    return summary


def normalize_category(filename_stem: str) -> str:
    """Convert filename stem to a readable category string."""
    return filename_stem.upper()


def find_tay_nung_dirs(root: Path) -> list[tuple[str, Path]]:
    """
    Walk root looking for Tày-Nùng (or similar) top-level folder.
    Returns list of (site_name, json_file) tuples.

    Note: macOS HFS+ uses NFD Unicode normalization for filenames, so we
    normalize to NFC before comparing to work correctly on both platforms.
    """
    import unicodedata
    TAY_KEYWORDS = ("tay", "t\u00e0y", "nung", "n\u00f9ng")

    entries = []
    for group_dir in sorted(root.iterdir()):
        if not group_dir.is_dir():
            continue
        # Normalize to NFC so macOS NFD filenames compare correctly
        name_nfc = unicodedata.normalize("NFC", group_dir.name).lower()
        is_tay = any(k in name_nfc for k in TAY_KEYWORDS)
        if not is_tay:
            continue
        # Found the Tày-Nùng group folder
        for site_dir in sorted(group_dir.iterdir()):
            if not site_dir.is_dir():
                continue
            site_name = site_dir.name  # e.g. "VOV"
            for json_file in sorted(site_dir.glob("*.json")):
                entries.append((site_name, json_file))
    return entries


# ── Main ───────────────────────────────────────────────────────────────────────

def tay_nung_parser(data_dir: str = "data", output_dir: str = "."):
    root = Path(data_dir)
    if not root.exists():
        print(f"[ERROR] Directory '{data_dir}' does not exist.")
        sys.exit(1)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not HAS_TQDM:
        print("[INFO] tqdm not found — install with: pip install tqdm\n")

    print(f"Scanning: {root.resolve()}")
    file_entries = find_tay_nung_dirs(root)

    if not file_entries:
        print("[WARNING] No Tày-Nùng JSON files found. Check directory structure.")
        return

    total_files = len(file_entries)
    print(f"Found {total_files} category file(s).\n")

    all_records = []
    skipped_files = 0
    skipped_records = 0

    for site_name, json_file in tqdm(file_entries, desc="Parsing files", unit="file"):
        category = normalize_category(json_file.stem)

        try:
            raw = json.loads(json_file.read_text(encoding="utf-8", errors="replace"))
        except json.JSONDecodeError as e:
            print(f"\n[WARN] Cannot parse {json_file.name}: {e}")
            skipped_files += 1
            continue

        if not isinstance(raw, list):
            # Some files may wrap the list in a dict — try to unwrap
            if isinstance(raw, dict):
                raw = next((v for v in raw.values() if isinstance(v, list)), [])
            else:
                skipped_files += 1
                continue

        for item in raw:
            if not isinstance(item, dict):
                skipped_records += 1
                continue

            content = item.get("content", "").strip()
            title   = item.get("title", "").strip()
            url     = item.get("url", "").strip()

            # Skip empty/useless records
            if not url and not title and not content:
                skipped_records += 1
                continue

            summary = item.get("summary", "").strip() or extract_summary(content)

            record = {
                "site":     site_name,
                "category": category,
                "url":      url,
                "title":    title,
                "summary":  summary,
                "content":  content,
            }
            all_records.append(record)

    out_file = output_path / "tay_nung.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)

    print(f"\n{len(all_records)} records → {out_file}")
    if skipped_files:
        print(f"{skipped_files} file(s) skipped (parse error)")
    if skipped_records:
        print(f"{skipped_records} record(s) skipped (empty or invalid)")
    print("Done!")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Parse Tày-Nùng VOV JSON corpus into a single tay_nung.json")
    ap.add_argument("--data-dir",    default="data", help="Root data directory (default: data/)")
    ap.add_argument("--output-dir",  default=".",    help="Output directory (default: current dir)")
    args = ap.parse_args()

    tay_nung_parser(data_dir=args.data_dir, output_dir=args.output_dir)