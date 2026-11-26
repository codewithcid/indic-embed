"""Native-word -> English-word lexicon for code-mixing augmentation, induced from the teacher's embeddings.

Native and romanized paragraphs in Sangraha are transliterated word by word, so when the word counts match, word i of
the romanized text is the romanization of word i of the native text. Native words are matched to English words
(from the English training paragraphs) with margin-based mutual nearest neighbours (Artetxe & Schwenk, 2019) over
multilingual-e5-small embeddings. Output: data/train/lexicon.json {lang: {native_word: english_word}}

    python build_lexicon.py --per-lang 6000 --top 1000
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import torch
import unicodedata
from sentence_transformers import SentenceTransformer
from wordfreq import zipf_frequency

DATA = Path("data/train")
EN_WORD = re.compile(r"^[a-z]{3,}$")


def _strip(w: str) -> str:
    """Strip punctuation at both ends; Indic vowel signs are not \w, so a regex would cut them off."""
    a, b = 0, len(w)
    while a < b and unicodedata.category(w[a])[0] in "PSZ":
        a += 1
    while b > a and unicodedata.category(w[b - 1])[0] in "PSZ":
        b -= 1
    return w[a:b]


def words(text: str) -> list[str]:
    return [_strip(w) for w in text.split()]


@torch.no_grad()
def embed(model, texts: list[str]) -> torch.Tensor:
    return model.encode(["query: " + t for t in texts], batch_size=512, convert_to_tensor=True,
                        normalize_embeddings=True, show_progress_bar=False).float()


def knn_mean(a: torch.Tensor, b: torch.Tensor, k: int) -> torch.Tensor:
    return torch.cat([(a[i:i + 4096] @ b.T).topk(k, dim=1).values.mean(1) for i in range(0, len(a), 4096)])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--per-lang", type=int, default=6000, help="most frequent native words per language")
    p.add_argument("--skip-top", type=int, default=150, help="most frequent native words treated as function words")
    p.add_argument("--english", type=int, default=40000)
    p.add_argument("--top", type=int, default=1000, help="mutual nearest pairs kept per language, by margin")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--min-zipf", type=float, default=2.5, help="wordfreq English frequency for the English side")
    args = p.parse_args()
    pairs = [json.loads(line) for line in (DATA / "pairs.jsonl").open(encoding="utf-8")]
    english = [json.loads(line)["text"] for line in (DATA / "english.jsonl").open(encoding="utf-8")]

    model = SentenceTransformer("intfloat/multilingual-e5-small", device="cuda")
    model.half()
    # the English paragraphs contain some Malay/Tagalog/etc.; keep words that are common in English
    en_counts = Counter(w for t in english for w in (x.lower() for x in words(t)) if EN_WORD.match(w))
    en_vocab = [w for w, _ in en_counts.most_common() if zipf_frequency(w, "en") >= args.min_zipf][:args.english]
    en = embed(model, en_vocab)

    counts: dict[str, Counter] = defaultdict(Counter)
    for r in pairs:
        n, ro = words(r["native"]), words(r["roman"])
        if len(n) == len(ro):
            counts[r["lang"]].update(w for w in n if w and not w.isdigit())
    lexicon = {}
    for lang, c in sorted(counts.items()):
        native = [w for w, _ in c.most_common(args.per_lang + args.skip_top)[args.skip_top:]]
        x = embed(model, native)
        x_knn = knn_mean(x, en, args.k)
        en_knn_lang = knn_mean(en, x, args.k)
        sims = x @ en.T
        best = sims.argmax(1)
        back = sims.argmax(0)  # nearest native word for each English word
        margin = sims[torch.arange(len(x)), best] / ((x_knn + en_knn_lang[best]) / 2)
        mutual = [i for i in margin.argsort(descending=True).tolist() if back[best[i]] == i]
        keep = {native[i]: en_vocab[best[i]] for i in mutual[:args.top]}
        lexicon[lang] = keep
        print(f"{lang}: {len(keep)} of {len(native)} words (margin >= {margin[mutual[:args.top][-1]]:.3f})", flush=True)
    (DATA / "lexicon.json").write_text(json.dumps(lexicon, ensure_ascii=False, indent=0), encoding="utf-8")


if __name__ == "__main__":
    main()
