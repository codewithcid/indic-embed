"""Check trimmed models against the original e5-small.

Per source text set: share of texts whose token pieces are identical, new <unk> tokens (should be none), and the
cosine similarity between the original and trimmed embeddings (1.0 where segmentation is identical).

    python verify_trim.py models/e5-small-indic-32k models/e5-small-indic-48k models/e5-small-indic-64k
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from huggingface_hub import HfApi, hf_hub_download
from sentence_transformers import SentenceTransformer
from tokenizers import Tokenizer

BASE = "intfloat/multilingual-e5-small"
SIB = {"hin_Deva": "hi", "ben_Beng": "bn", "tam_Taml": "ta", "tel_Telu": "te", "mal_Mlym": "ml", "mar_Deva": "mr",
       "guj_Gujr": "gu", "kan_Knda": "kn", "pan_Guru": "pa", "ory_Orya": "or", "asm_Beng": "as", "urd_Arab": "ur",
       "npi_Deva": "ne", "eng_Latn": "en"}


def text_sets() -> dict[str, list[str]]:
    sets = {}
    files = HfApi().list_repo_files("Davlan/sib200", repo_type="dataset")
    for code, name in SIB.items():
        f = next(x for x in files if x == f"data/{code}/test.tsv")
        with open(hf_hub_download("Davlan/sib200", f, repo_type="dataset"), encoding="utf-8", newline="") as fh:
            sets[name] = [r["text"] for r in csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE) if r.get("text")]
    roman = []
    for tsv in sorted(Path("data/dakshina/dakshina_dataset_v1.0").glob("*/romanized/*.romanized.rejoined.tsv")):
        lines = [ln.split("\t") for ln in tsv.read_text(encoding="utf-8").splitlines()[:300]]
        roman += [p[1] for p in lines if len(p) == 2]
    sets["romanized"] = roman
    return sets


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    base_dir = Path(hf_hub_download(BASE, "tokenizer.json")).parent
    base_tok = Tokenizer.from_file(str(base_dir / "tokenizer.json"))
    base_model = SentenceTransformer(BASE, device="cpu")
    sets = text_sets()
    base_emb = {k: base_model.encode(["query: " + t for t in v[:200]], normalize_embeddings=True) for k, v in sets.items()}
    for model_dir in sys.argv[1:]:
        tok = Tokenizer.from_file(str(Path(model_dir) / "tokenizer.json"))
        model = SentenceTransformer(model_dir, device="cpu")
        unk = tok.token_to_id("<unk>")
        print(f"== {model_dir} (vocab {tok.get_vocab_size():,})")
        print(f"   {'source':10} {'identical':>10} {'new <unk>':>10} {'cosine(min/mean)':>18}")
        for name, texts in sets.items():
            a_enc = base_tok.encode_batch(texts, add_special_tokens=False)
            b_enc = tok.encode_batch(texts, add_special_tokens=False)
            same = sum(x.tokens == y.tokens for x, y in zip(a_enc, b_enc))
            # Count unknowns by id on both sides: the token string of an unknown is the original character.
            base_unk = sum(e.ids.count(base_tok.token_to_id("<unk>")) for e in a_enc)
            new_unk = sum(e.ids.count(unk) for e in b_enc) - base_unk
            emb = model.encode(["query: " + t for t in texts[:200]], normalize_embeddings=True)
            cos = (emb * base_emb[name]).sum(1)
            print(f"   {name:10} {same / len(texts):>9.1%} {new_unk:>10} {cos.min():>9.3f}/{cos.mean():.4f}")


if __name__ == "__main__":
    main()
