"""Romanized and code-mixed Indic evaluation, which MTEB(Indic, v1) barely covers.

Tasks:
- dakshina-xscript: Google's Dakshina romanized sentences (CC BY-SA 4.0). For each language, every romanized
  sentence is a query and the native-script originals are the corpus (and the other way round). Recall@1 and
  MRR@10. Measures whether a model maps "aapka khata band" and "आपका खाता बंद" to the same place.
- tanglish-sts: vishnu-n/TanglishSTS (CC BY 4.0), 325 Tamil-English code-mixed pairs scored 0-5. Spearman.

    python eval_codemix.py intfloat/multilingual-e5-small --max-per-lang 2000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).parent
DAKSHINA = ROOT / "data" / "dakshina"
LANGS = ["bn", "gu", "hi", "kn", "ml", "mr", "pa", "ta", "te", "ur"]

# How each model expects inputs. Symmetric tasks use the query form on both sides.
PROMPTS = {
    "intfloat/multilingual-e5-small": ("query: ", "query: "),
    "intfloat/multilingual-e5-base": ("query: ", "query: "),
    "intfloat/multilingual-e5-large": ("query: ", "query: "),
}


def load_dakshina(lang: str, max_pairs: int) -> list[tuple[str, str]]:
    files = sorted(DAKSHINA.rglob(f"{lang}/romanized/{lang}.romanized.rejoined.tsv"))
    if not files:
        raise FileNotFoundError(f"Dakshina romanized file for {lang} not found under {DAKSHINA}")
    pairs, seen = [], set()
    for line in files[0].read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) != 2:
            continue
        native, roman = parts[0].strip(), parts[1].strip()
        if native and roman and native not in seen and len(native.split()) >= 3:
            seen.add(native)
            pairs.append((native, roman))
    rng = np.random.default_rng(0)
    if len(pairs) > max_pairs:
        pairs = [pairs[i] for i in sorted(rng.choice(len(pairs), max_pairs, replace=False))]
    return pairs


def encode(model: SentenceTransformer, texts: list[str], prefix: str, batch_size: int) -> np.ndarray:
    return model.encode([prefix + t for t in texts], batch_size=batch_size, normalize_embeddings=True,
                        convert_to_numpy=True, show_progress_bar=False)


def retrieval_scores(q: np.ndarray, d: np.ndarray) -> dict:
    sims = q @ d.T
    ranks = (sims > np.diag(sims)[:, None]).sum(axis=1)  # gold doc for query i is doc i
    return {"recall@1": float((ranks == 0).mean()), "mrr@10": float(np.where(ranks < 10, 1.0 / (ranks + 1), 0).mean())}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--like", help="for a local model: the Hub model whose prompts to use")
    p.add_argument("--max-per-lang", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--out", default="results/codemix")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(args.model, device=device, trust_remote_code=False)
    if device == "cuda":
        model.half()
    qp, dp = PROMPTS.get(args.like or args.model, ("", ""))
    t0 = time.time()
    out = {"model": f"local/{Path(args.model).name}" if args.like else args.model, "dakshina": {}, "tanglish_sts": None}

    for lang in LANGS:
        pairs = load_dakshina(lang, args.max_per_lang)
        native = encode(model, [n for n, _ in pairs], dp, args.batch_size)
        roman = encode(model, [r for _, r in pairs], qp, args.batch_size)
        out["dakshina"][lang] = {
            "n": len(pairs),
            "roman_to_native": retrieval_scores(roman, native),
            "native_to_roman": retrieval_scores(native, roman),
        }
        r = out["dakshina"][lang]
        print(f"  dakshina {lang}: n={len(pairs)} R@1 roman->native {r['roman_to_native']['recall@1']:.3f} "
              f"native->roman {r['native_to_roman']['recall@1']:.3f}", flush=True)

    from huggingface_hub import hf_hub_download

    rows = [json.loads(line) for line in open(hf_hub_download("vishnu-n/TanglishSTS", "tanglish_sts.jsonl",
                                                               repo_type="dataset"), encoding="utf-8")]
    s1, s2, score = "s1", "s2", "human_score"
    a = encode(model, [r[s1] for r in rows], qp, args.batch_size)
    b = encode(model, [r[s2] for r in rows], qp, args.batch_size)
    out["tanglish_sts"] = {"n": len(rows), "spearman": float(spearmanr((a * b).sum(1), [float(r[score]) for r in rows])[0])}
    print(f"  tanglish-sts spearman {out['tanglish_sts']['spearman']:.3f} (n={len(rows)})")

    r2n = np.mean([v["roman_to_native"]["recall@1"] for v in out["dakshina"].values()])
    n2r = np.mean([v["native_to_roman"]["recall@1"] for v in out["dakshina"].values()])
    out["summary"] = {"dakshina_roman_to_native_r1": float(r2n), "dakshina_native_to_roman_r1": float(n2r),
                      "tanglish_sts_spearman": out["tanglish_sts"]["spearman"], "seconds": time.time() - t0}
    print(f"{args.model}: dakshina R@1 roman->native {r2n:.3f}, native->roman {n2r:.3f}; "
          f"tanglish-sts {out['tanglish_sts']['spearman']:.3f}")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / f"{('local__' + Path(args.model).name) if args.like else args.model.replace('/', '__')}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
