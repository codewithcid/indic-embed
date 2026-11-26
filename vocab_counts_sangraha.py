"""Add token counts from AI4Bharat's Sangraha (CC BY 4.0): Verified web text per language and the romanized
(_Latn) part of Sangraha Synthetic. Reads half the documents from the first shard and half from a middle one, so
one crawl source does not dominate, without downloading whole datasets.

    python vocab_counts_sangraha.py --docs 20000
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi
from transformers import AutoTokenizer

VERIFIED = ["hin", "ben", "tam", "tel", "mal", "mar", "guj", "kan", "pan", "ori", "asm", "urd", "nep", "eng"]
ROMANIZED = ["hin_Latn", "ben_Latn", "tam_Latn", "tel_Latn", "mal_Latn", "mar_Latn", "guj_Latn", "kan_Latn",
             "pan_Latn", "ory_Latn", "asm_Latn", "urd_Latn", "npi_Latn"]


def shard_names(files: list[str], data_dir: str) -> list[str]:
    # verified/<lang>/data-N.parquet and synthetic/<lang>/wiki_<lang>_NNNN_of_NNNN.parquet: sort by first number
    shards = sorted(
        (f for f in files if f.startswith(data_dir + "/") and f.endswith(".parquet")),
        key=lambda f: int(re.findall(r"\d+", f.rsplit("/", 1)[1])[0]),
    )
    return [shards[0], shards[len(shards) // 2]] if len(shards) > 1 else shards


def texts(files: list[str], data_dir: str, docs: int):
    for shard in shard_names(files, data_dir):
        ds = load_dataset("ai4bharat/sangraha", data_files=shard, split="train", streaming=True)
        for i, row in enumerate(ds):
            if i >= docs // 2:
                break
            for para in row["text"].split("\n"):
                if len(para) > 20:
                    yield para[:2000]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="intfloat/multilingual-e5-small")
    p.add_argument("--docs", type=int, default=20000)
    args = p.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model)
    files = [s.rfilename for s in HfApi().dataset_info("ai4bharat/sangraha").siblings]
    out = Path("results/vocab")
    jobs = [(f"verified/{lang}", f"counts_sangraha_{lang}.json") for lang in VERIFIED]
    jobs += [(f"synthetic/{lang}", f"counts_sangraha_{lang}.json") for lang in ROMANIZED]
    for data_dir, name in jobs:
        f = out / name
        if f.exists():
            continue
        c, n, batch = Counter(), 0, []
        for t in texts(files, data_dir, args.docs):
            batch.append(t)
            if len(batch) == 256:
                for ids in tok(batch, add_special_tokens=False)["input_ids"]:
                    c.update(ids)
                    n += len(ids)
                batch = []
        if batch:
            for ids in tok(batch, add_special_tokens=False)["input_ids"]:
                c.update(ids)
                n += len(ids)
        f.write_text(json.dumps(dict(c)), encoding="utf-8")
        print(f"{data_dir}: {n:,} tokens, {len(c):,} distinct", flush=True)


if __name__ == "__main__":
    main()
