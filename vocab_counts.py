"""Count how often each token of multilingual-e5-small's tokenizer is used on Indic, English and romanized text.

Text: the first N Wikipedia articles (wikimedia/wikipedia, CC BY-SA) per language, streamed, plus Dakshina's
romanized sentences (CC BY-SA 4.0). Output: results/vocab/counts_<lang>.json (token id -> count).

    python vocab_counts.py --articles 20000
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

WIKI_LANGS = ["hi", "bn", "ta", "te", "ml", "mr", "gu", "kn", "pa", "or", "as", "ur", "ne", "en"]
DAKSHINA = Path("data/dakshina/dakshina_dataset_v1.0")


def count(tok, texts, counter: Counter) -> int:
    n = 0
    batch = []
    for t in texts:
        batch.append(t)
        if len(batch) == 256:
            for ids in tok(batch, add_special_tokens=False)["input_ids"]:
                counter.update(ids)
                n += len(ids)
            batch = []
    if batch:
        for ids in tok(batch, add_special_tokens=False)["input_ids"]:
            counter.update(ids)
            n += len(ids)
    return n


def paragraphs(lang: str, articles: int):
    ds = load_dataset("wikimedia/wikipedia", f"20231101.{lang}", split="train", streaming=True)
    for i, row in enumerate(ds):
        if i >= articles:
            break
        for para in row["text"].split("\n"):
            if len(para) > 20:
                yield para[:2000]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="intfloat/multilingual-e5-small")
    p.add_argument("--articles", type=int, default=20000)
    args = p.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model)
    out = Path("results/vocab")
    out.mkdir(parents=True, exist_ok=True)

    for lang in WIKI_LANGS:
        f = out / f"counts_wiki_{lang}.json"
        if f.exists():
            continue
        c = Counter()
        n = count(tok, paragraphs(lang, args.articles), c)
        f.write_text(json.dumps(dict(c)), encoding="utf-8")
        print(f"wiki {lang}: {n:,} tokens, {len(c):,} distinct", flush=True)

    c = Counter()
    lines = []
    for tsv in sorted(DAKSHINA.glob("*/romanized/*.romanized.rejoined.tsv")):
        for line in tsv.read_text(encoding="utf-8").splitlines():
            parts = line.split("\t")
            if len(parts) == 2:
                lines.append(parts[1])
    n = count(tok, lines, c)
    (out / "counts_romanized.json").write_text(json.dumps(dict(c)), encoding="utf-8")
    print(f"romanized (Dakshina): {n:,} tokens, {len(c):,} distinct", flush=True)


if __name__ == "__main__":
    main()
