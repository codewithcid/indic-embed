"""Augmentations that make the clean Sangraha romanization look like what people type.

code_mix(): word i of a romanized paragraph is the romanization of word i of the native one (Sangraha transliterates
word by word), so words found in the induced native->English lexicon (build_lexicon.py) can be swapped for English,
as in Hinglish/Tanglish ("naan office ku poren").
casual(): spelling variation (vowel length, aspiration, doubled consonants, zh, ...).
"""

from __future__ import annotations

import random
import re
import unicodedata

DRAVIDIAN = {"tam_Taml", "mal_Mlym", "kan_Knda", "tel_Telu"}


def split_punct(w: str) -> tuple[str, str, str]:
    a, b = 0, len(w)
    while a < b and unicodedata.category(w[a])[0] in "PSZ":
        a += 1
    while b > a and unicodedata.category(w[b - 1])[0] in "PSZ":
        b -= 1
    return w[:a], w[a:b], w[b:]


def code_mix(native: str, roman: str, lex: dict, rng: random.Random, p: float = 0.6) -> str:
    n, r = native.split(), roman.split()
    if len(n) != len(r):
        return roman
    out = []
    for nw, rw in zip(n, r):
        en = lex.get(split_punct(nw)[1])
        if en and rng.random() < p:
            pre, _, post = split_punct(rw)
            rw = pre + en + post
        out.append(rw)
    return " ".join(out)


def casual(text: str, rng: random.Random, lang: str = "") -> str:
    """Spelling variation people use when typing romanized Indic text."""
    t = text.lower() if rng.random() < 0.5 else text
    rules = [("aa", "a"), ("ee", "i"), ("oo", "u"), ("ii", "i"), ("uu", "u"), ("bh", "b"), ("dh", "d"), ("th", "t"),
             ("kh", "k"), ("gh", "g"), ("jh", "j"), ("ph", "f"), ("ack ", "ek "), ("haye", "hai"), ("v", "w"),
             ("sh", "s"), ("ai", "ei")]
    if lang in DRAVIDIAN:
        rules += [("zh", rng.choice(["l", "z"]))]
    if lang == "tam_Taml":
        rules += [("ch", "s")]
    for a, b in rules:
        if rng.random() < 0.35:
            t = t.replace(a, b)
    if rng.random() < 0.35:  # kk -> k, pp -> p, ...
        t = re.sub(r"([bcdfgjklmnprstvBCDFGJKLMNPRSTV])\1", r"\1", t)
    return t


def aligned_crop(native: str, roman: str, rng: random.Random, lo: int = 2, hi: int = 8) -> tuple[str, str] | None:
    """The same 2-8 word window of a word-aligned native/romanized pair: short, query-like text."""
    n, r = native.split(), roman.split()
    if len(n) != len(r) or len(n) < lo:
        return None
    k = rng.randint(lo, min(hi, len(n)))
    s = rng.randint(0, len(n) - k)
    return " ".join(n[s:s + k]), " ".join(r[s:s + k])
