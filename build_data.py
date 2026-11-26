"""Phase A training data: aligned native/romanized paragraphs from Sangraha Synthetic (CC BY 4.0) + English.

For each language, rows of the native and romanized shards are aligned by doc_id, split into paragraphs (the
paragraph counts match), and kept when the native paragraph is 20-600 characters. Half the pairs come from the
first shard and half from a middle one. A held-out dev set (per language) is used only to monitor training;
Dakshina and TanglishSTS, the real evaluation sets, are never used.

Output: data/train/pairs.jsonl ({"lang", "native", "roman"}), data/train/dev.jsonl, data/train/english.jsonl

    python build_data.py --pairs-per-lang 20000
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset

LANGS = ["hin_Deva", "ben_Beng", "tam_Taml", "tel_Telu", "mal_Mlym", "mar_Deva", "guj_Gujr", "kan_Knda",
         "pan_Guru", "ory_Orya", "asm_Beng", "urd_Arab", "npi_Deva"]
SHARDS = ["0000", "0032"]
OUT = Path("data/train")


def shard_rows(cfg: str, shard: str):
    f = f"synthetic/{cfg}/wiki_{cfg}_{shard}_of_0063.parquet"
    return iter(load_dataset("ai4bharat/sangraha", data_files=f, split="train", streaming=True))


def paragraphs(text: str) -> list[str]:
    return [p.strip() for p in text.split("\n") if p.strip()]


def pairs_for(lang: str, want: int):
    roman_cfg = lang.split("_")[0] + "_Latn"
    for shard in SHARDS:
        got = 0
        for nat, rom in zip(shard_rows(lang, shard), shard_rows(roman_cfg, shard)):
            if nat["doc_id"] != rom["doc_id"]:
                raise ValueError(f"{lang} {shard}: rows out of alignment")
            pn, pr = paragraphs(nat["text"]), paragraphs(rom["text"])
            if len(pn) != len(pr):
                continue
            for a, b in zip(pn, pr):
                if 20 <= len(a) <= 600 and len(b) >= 10:
                    yield a, b
                    got += 1
            if got >= want // len(SHARDS):
                break


def english(n: int) -> list[str]:
    out = []
    ds = load_dataset("ai4bharat/sangraha", data_files="verified/eng/data-0.parquet", split="train", streaming=True)
    for row in ds:
        for p in paragraphs(row["text"]):
            if 20 <= len(p) <= 600:
                out.append(p)
        if len(out) >= n:
            break
    return out[:n]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pairs-per-lang", type=int, default=20000)
    p.add_argument("--dev-per-lang", type=int, default=500)
    p.add_argument("--english", type=int, default=30000)
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    train, dev = [], []
    for lang in LANGS:
        pairs = list(dict.fromkeys(pairs_for(lang, args.pairs_per_lang + args.dev_per_lang)))  # drop duplicates
        rng.shuffle(pairs)
        dev += [{"lang": lang, "native": a, "roman": b} for a, b in pairs[: args.dev_per_lang]]
        train += [{"lang": lang, "native": a, "roman": b} for a, b in pairs[args.dev_per_lang:]]
        print(f"{lang}: {len(pairs) - args.dev_per_lang} train, {args.dev_per_lang} dev", flush=True)
    rng.shuffle(train)
    with (OUT / "pairs.jsonl").open("w", encoding="utf-8") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (OUT / "dev.jsonl").open("w", encoding="utf-8") as f:
        for r in dev:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    eng = english(args.english)
    with (OUT / "english.jsonl").open("w", encoding="utf-8") as f:
        for t in eng:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
    print(f"total: {len(train)} train pairs, {len(dev)} dev pairs, {len(eng)} English paragraphs")


if __name__ == "__main__":
    main()
