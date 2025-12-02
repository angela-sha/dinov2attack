from __future__ import annotations
import argparse
import json
import csv
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Set

DEFAULT_STOPWORDS: Set[str] = {
    "a","about","above","after","again","against","all","am","an","and","any","are","aren't","as","at",
    "be","because","been","before","being","below","between","both","but","by",
    "can't","cannot","could","couldn't",
    "did","didn't","do","does","doesn't","doing","don't","down","during",
    "each",
    "few","for","from","further",
    "had","hadn't","has","hasn't","have","haven't","having","he","he'd","he'll","he's","her","here",
    "here's","hers","herself","him","himself","his","how","how's",
    "i","i'd","i'll","i'm","i've","if","in","into","is","isn't","it","it's","its","itself",
    "let's",
    "me","more","most","mustn't","my","myself",
    "no","nor","not",
    "of","off","on","once","only","or","other","ought","our","ours","ourselves","out","over","own",
    "same","she","she'd","she'll","she's","should","shouldn't","so","some","such",
    "than","that","that's","the","their","theirs","them","themselves","then","there","there's","these","they","they'd","they'll","they're","they've","this","those","through","to",
    "too",
    "under","until","up",
    "very",
    "was","wasn't","we","we'd","we'll","we're","we've","were","weren't","what","what's","when","when's","where","where's","which","while","who","who's","whom","why","why's","with","won't","would","wouldn't",
    "you","you'd","you'll","you're","you've","your","yours","yourself","yourselves"
}
DEFAULT_STOPWORDS.update({"'s", "n't", "'m", "'re", "'ve", "--", "``", "''"})
TOKEN_RE = re.compile(r"\b\w[\w']*\b", flags=re.UNICODE)

def extract_captions_from_coco_json(path: Path) -> List[str]:
    """Load captions from a COCO-format JSON file. Returns list of caption strings."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    captions = []
    # COCO standard: "annotations" is a list of objects with "caption"
    if isinstance(data, dict) and "annotations" in data:
        for ann in data["annotations"]:
            # annotation might be something else; guard against missing keys
            cap = ann.get("caption") or ann.get("text") or None
            if cap:
                captions.append(cap)
    else:
        if isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    captions.append(item)
                elif isinstance(item, dict):
                    cap = item.get("caption") or item.get("text")
                    if cap:
                        captions.append(cap)
    return captions

def extract_captions_from_text_file(path: Path) -> List[str]:
    """Read a plain text file where each line is a caption (or many captions)."""
    captions: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                captions.append(line)
    return captions

def iter_captions_from_paths(paths: Iterable[Path]) -> Iterable[str]:
    """Given file paths, yield captions detected from the file type."""
    for p in paths:
        if not p.exists():
            print(f"Warning: input path does not exist: {p}", file=sys.stderr)
            continue
        if p.suffix.lower() in {".json"}:
            yield from extract_captions_from_coco_json(p)
        else:
            # treat as text file
            yield from extract_captions_from_text_file(p)

def tokenize(text: str) -> List[str]:
    """Tokenize a caption into words. Lowercases, keeps contractions like don't as don't,
    but our stopword list will handle common contractions."""
    text = text.lower()
    tokens = TOKEN_RE.findall(text)
    # filter out tokens that are only numeric or single-character punctuation
    filtered = [t for t in tokens if not t.isnumeric()]
    return filtered

def build_stopword_set(custom_stopword_file: str | None, extra: Iterable[str] | None) -> Set[str]:
    stopset = set(DEFAULT_STOPWORDS)
    if extra:
        stopset.update(w.lower() for w in extra)
    if custom_stopword_file:
        p = Path(custom_stopword_file)
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    w = line.strip().lower()
                    if w:
                        stopset.add(w)
        else:
            print(f"Warning: custom stopword file not found: {custom_stopword_file}", file=sys.stderr)
    return stopset

def count_words(captions: Iterable[str], stopwords: Set[str], min_len: int=2, exclude_numeric: bool=True) -> Counter:
    cnt = Counter()
    for cap in captions:
        for token in tokenize(cap):
            if exclude_numeric and token.isnumeric():
                continue
            if len(token) < min_len:
                continue
            if token in stopwords:
                continue
            cnt[token] += 1
    return cnt

def write_csv(counter: Counter, out_path: Path):
    with out_path.open("w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["word","count"])
        for word, count in counter.most_common():
            writer.writerow([word, count])

def parse_args():
    p = argparse.ArgumentParser(description="Count word frequencies in MS COCO captions (or plain caption text).")
    p.add_argument("--inputs", "-i", nargs="+", required=True,
                   help="Input file(s). COCO JSON(s) or plain text files (one caption per line).")
    p.add_argument("--out", "-o", default="word_freq.csv", help="Output CSV file (default: word_freq.csv)")
    p.add_argument("--top", "-t", type=int, default=50, help="Print top N words to stdout (default 50)")
    p.add_argument("--stopwords", "-s", default=None,
                   help="Optional path to a custom stopword file (one word per line).")
    p.add_argument("--extra-stopwords", "-e", nargs="*", default=None,
                   help="Extra stopwords to add on top of the built-in list (space separated).")
    p.add_argument("--min-len", type=int, default=2, help="Minimum token length to keep (default 2)")
    p.add_argument("--exclude-numeric", action="store_true", help="Exclude numeric tokens")
    return p.parse_args()

def main():
    args = parse_args()
    input_paths = [Path(p) for p in args.inputs]

    stopwords = build_stopword_set(args.stopwords, args.extra_stopwords)

    print("Reading captions from:", ", ".join(str(p) for p in input_paths))
    captions_iter = iter_captions_from_paths(input_paths)

    print("Counting words...")
    counter = count_words(captions_iter, stopwords, min_len=args.min_len, exclude_numeric=args.exclude_numeric)

    out_path = Path(args.out)
    write_csv(counter, out_path)

    print(f"\nTop {args.top} words:")
    for i, (word, cnt) in enumerate(counter.most_common(args.top), start=1):
        print(f"{i:3}. {word:20} {cnt}")

if __name__ == "__main__":
    main()
