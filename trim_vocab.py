"""Trim multilingual-e5-small's vocabulary to the pieces Indian languages, English and romanized text use.

Keeps: the special tokens; every single-character piece of the target scripts and every non-letter
single-character piece (digits, punctuation, symbols), so the trimmed Unigram model never needs `<unk>` where
the original did not; then the most used pieces per source (results/vocab/counts_*.json), chosen so every source
gets the same coverage, until the budget is filled. Pieces keep their scores and relative order, and the
embedding rows are gathered by the old ids. All other weights are unchanged.

    python trim_vocab.py --budget 65536 --out models/e5-small-indic-64k
"""

from __future__ import annotations

import argparse
import json
import shutil
import unicodedata
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModel

BASE = "intfloat/multilingual-e5-small"
TARGET_SCRIPTS = {"DEVANAGARI", "BENGALI", "GURMUKHI", "GUJARATI", "ORIYA", "TAMIL", "TELUGU", "KANNADA",
                  "MALAYALAM", "ARABIC", "LATIN"}
METASPACE = chr(0x2581)


def script(ch: str) -> str:
    try:
        return unicodedata.name(ch).split()[0]
    except ValueError:
        return "UNNAMED"


def required_ids(vocab: list) -> set[int]:
    keep = {0, 1, 2, 3, len(vocab) - 1}  # <s> <pad> </s> <unk> ... <mask>
    for i, (piece, _) in enumerate(vocab):
        core = piece[1:] if piece.startswith(METASPACE) else piece
        if piece == METASPACE:
            keep.add(i)
        elif len(core) == 1 and (script(core) in TARGET_SCRIPTS or not unicodedata.category(core).startswith("L")):
            keep.add(i)
    return keep


def top_for(counts: dict[int, int], coverage: float) -> set[int]:
    total, acc, out = sum(counts.values()), 0, set()
    for tid, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        if acc >= coverage * total:
            break
        out.add(tid)
        acc += n
    return out


# counts_<source>.json -> language group. Each file is normalized to frequencies first, so a large corpus does not
# outweigh a small one within a group, and every group (language, English, romanized) gets the same coverage.
GROUPS = {
    "wiki_hi": "hi", "sangraha_hin": "hi", "wiki_bn": "bn", "sangraha_ben": "bn", "wiki_ta": "ta", "sangraha_tam": "ta",
    "wiki_te": "te", "sangraha_tel": "te", "wiki_ml": "ml", "sangraha_mal": "ml", "wiki_mr": "mr", "sangraha_mar": "mr",
    "wiki_gu": "gu", "sangraha_guj": "gu", "wiki_kn": "kn", "sangraha_kan": "kn", "wiki_pa": "pa", "sangraha_pan": "pa",
    "wiki_or": "or", "sangraha_ori": "or", "wiki_as": "as", "sangraha_asm": "as", "wiki_ur": "ur", "sangraha_urd": "ur",
    "wiki_ne": "ne", "sangraha_nep": "ne", "wiki_en": "en", "sangraha_eng": "en",
}


def load_groups(counts_dir: Path) -> list[dict[int, float]]:
    groups: dict[str, dict[int, float]] = {}
    for p in sorted(counts_dir.glob("counts_*.json")):
        source = p.stem.removeprefix("counts_")
        group = "romanized" if source.endswith("_Latn") else GROUPS[source]
        raw = {int(k): v for k, v in json.loads(p.read_text(encoding="utf-8")).items()}
        total = sum(raw.values())
        g = groups.setdefault(group, {})
        for tid, n in raw.items():
            g[tid] = g.get(tid, 0.0) + n / total
    return list(groups.values())


def select(vocab: list, budget: int, counts_dir: Path) -> tuple[list[int], float]:
    counts = load_groups(counts_dir)
    base = required_ids(vocab)
    lo, hi = 0.5, 1.0
    for _ in range(40):  # largest balanced coverage that fits the budget
        mid = (lo + hi) / 2
        n = len(base | set().union(*(top_for(c, mid) for c in counts)))
        lo, hi = (mid, hi) if n <= budget else (lo, mid)
    keep = base | set().union(*(top_for(c, lo) for c in counts))
    return sorted(keep), lo


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--budget", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--counts", default="results/vocab")
    args = p.parse_args()

    src = Path(snapshot_download(BASE, allow_patterns=["*.json", "*.safetensors", "1_Pooling/*", "README.md"]))
    tj = json.loads((src / "tokenizer.json").read_text(encoding="utf-8"))
    vocab = tj["model"]["vocab"]
    old_ids, coverage = select(vocab, args.budget, Path(args.counts))
    new_id = {old: new for new, old in enumerate(old_ids)}

    tj["model"]["vocab"] = [vocab[i] for i in old_ids]
    tj["model"]["unk_id"] = new_id[tj["model"]["unk_id"]]
    for tok in tj["added_tokens"]:
        tok["id"] = new_id[tok["id"]]
    for spec in (tj.get("post_processor") or {}).get("special_tokens", {}).values():
        spec["ids"] = [new_id[i] for i in spec["ids"]]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tokenizer.json").write_text(json.dumps(tj, ensure_ascii=False), encoding="utf-8")
    for f in ["tokenizer_config.json", "special_tokens_map.json", "modules.json", "sentence_bert_config.json"]:
        shutil.copy(src / f, out / f)
    (out / "1_Pooling").mkdir(exist_ok=True)
    shutil.copy(src / "1_Pooling" / "config.json", out / "1_Pooling" / "config.json")

    model = AutoModel.from_pretrained(src)
    old = model.get_input_embeddings().weight.data
    emb = torch.nn.Embedding(len(old_ids), old.shape[1], padding_idx=new_id[model.config.pad_token_id])
    emb.weight.data = old[torch.tensor(old_ids)].clone()
    model.set_input_embeddings(emb)
    model.config.vocab_size = len(old_ids)
    model.config.pad_token_id = new_id[model.config.pad_token_id]
    model.save_pretrained(out)

    n_params = sum(t.numel() for t in model.parameters())
    meta = {"base": BASE, "budget": args.budget, "vocab_size": len(old_ids), "balanced_coverage": coverage,
            "parameters": n_params, "old_ids": old_ids}
    (out / "trim_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    print(f"{out}: vocab {len(old_ids):,} (balanced coverage {coverage:.4%}), parameters {n_params / 1e6:.1f}M")


if __name__ == "__main__":
    main()
