"""Phase A: teach the trimmed e5-small to embed romanized Indic text where the original embeds the native text.

Multilingual knowledge distillation (Reimers & Gurevych, 2020) with the original multilingual-e5-small as the
frozen teacher. For each aligned pair (native, romanized) the student is trained so that
    S(native) ~ T(native),  S(romanized) ~ T(native),  S(casual romanized) ~ T(native)
and, with --mix (data/train/lexicon.json from build_lexicon.py), S(code-mixed casual romanized) ~ T(native),
where some words are swapped for English as in Hinglish/Tanglish. S(english) ~ T(english) keeps English where it
was. Loss: 1 - cosine. Both e5 prefixes are used.

    python train_phase_a.py --precompute       # teacher embeddings, once
    python train_phase_a.py --epochs 1 --batch-size 64
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from augment import casual, code_mix

DATA = Path("data/train")
TEACHER = "intfloat/multilingual-e5-small"
STUDENT = "models/e5-small-indic-64k"
PREFIXES = ["query: ", "passage: "]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@torch.no_grad()
def precompute() -> None:
    teacher = SentenceTransformer(TEACHER, device="cuda")
    teacher.half()
    pairs = read_jsonl(DATA / "pairs.jsonl") + read_jsonl(DATA / "dev.jsonl")
    english = [r["text"] for r in read_jsonl(DATA / "english.jsonl")]
    for name, texts in [("native", [p["native"] for p in pairs]), ("english", english)]:
        for i, prefix in enumerate(PREFIXES):
            emb = teacher.encode([prefix + t for t in texts], batch_size=256, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False)
            np.save(DATA / f"teacher_{name}_{i}.npy", emb.astype(np.float16))
            print(f"teacher {name} prefix {prefix!r}: {emb.shape}", flush=True)


def embed(model: SentenceTransformer, texts: list[str]) -> torch.Tensor:
    feats = model.tokenize(texts)
    feats = {k: v.to(model.device) if torch.is_tensor(v) else v for k, v in feats.items()}
    return F.normalize(model(feats)["sentence_embedding"], dim=-1)


@torch.no_grad()
def evaluate(model: SentenceTransformer, dev: list[dict], t_dev: np.ndarray) -> dict:
    model.eval()
    out = {"r1": [], "drift": []}
    for lang in sorted({r["lang"] for r in dev}):
        idx = [i for i, r in enumerate(dev) if r["lang"] == lang]
        nat = torch.cat([embed(model, ["passage: " + dev[i]["native"] for i in idx[j:j + 128]])
                         for j in range(0, len(idx), 128)])
        rom = torch.cat([embed(model, ["query: " + dev[i]["roman"] for i in idx[j:j + 128]])
                         for j in range(0, len(idx), 128)])
        sims = rom @ nat.T
        out["r1"].append((sims.argmax(1) == torch.arange(len(idx), device=sims.device)).float().mean().item())
        teacher = torch.tensor(t_dev[idx], device=nat.device, dtype=nat.dtype)
        out["drift"].append((nat * teacher).sum(1).mean().item())
    model.train()
    return {"roman_to_native_r1": float(np.mean(out["r1"])), "native_cos_to_teacher": float(np.mean(out["drift"]))}


def train(args) -> None:
    rng = random.Random(0)
    torch.manual_seed(0)
    pairs = read_jsonl(DATA / "pairs.jsonl")
    dev = read_jsonl(DATA / "dev.jsonl")
    english = [r["text"] for r in read_jsonl(DATA / "english.jsonl")]
    t_nat = [np.load(DATA / f"teacher_native_{i}.npy") for i in range(2)]
    t_eng = [np.load(DATA / f"teacher_english_{i}.npy") for i in range(2)]
    n_train = len(pairs)
    t_dev = t_nat[1][n_train:]  # dev rows follow the train rows; "passage: " prefix, as in evaluate()

    lexicon = json.loads((DATA / "lexicon.json").read_text(encoding="utf-8")) if args.mix else None
    model = SentenceTransformer(STUDENT, device="cuda")
    model.max_seq_length = args.max_len
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * (n_train // args.batch_size)
    warmup = int(0.05 * steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * max(0.0, (steps - s) / max(1, steps - warmup)))
    scaler = torch.amp.GradScaler("cuda")
    out_dir = Path(args.out)
    print(f"{n_train} pairs, {steps} steps, warmup {warmup}; before: {evaluate(model, dev, t_dev)}", flush=True)

    best, step, t0 = -1.0, 0, time.time()
    for epoch in range(args.epochs):
        order = list(range(n_train))
        rng.shuffle(order)
        for b in range(0, n_train - args.batch_size + 1, args.batch_size):
            idx = order[b:b + args.batch_size]
            p = rng.randrange(2)
            prefix = PREFIXES[p]
            target = torch.tensor(t_nat[p][idx], device="cuda", dtype=torch.float32)
            nat = [prefix + pairs[i]["native"] for i in idx]
            rom = [prefix + pairs[i]["roman"] for i in idx]
            cas = [prefix + casual(pairs[i]["roman"], rng, pairs[i]["lang"]) for i in idx]
            views = [(nat, target), (rom, target), (cas, target)]
            if lexicon:
                mix = [code_mix(pairs[i]["native"], pairs[i]["roman"], lexicon[pairs[i]["lang"]], rng, args.mix_p)
                       for i in idx]
                views.append(([prefix + (casual(m, rng, pairs[i]["lang"]) if rng.random() < 0.5 else m)
                               for m, i in zip(mix, idx)], target))
            e_idx = rng.sample(range(len(english)), args.batch_size // 4)
            eng = [prefix + english[i] for i in e_idx]
            e_target = torch.tensor(t_eng[p][e_idx], device="cuda", dtype=torch.float32)
            # the loss is a per-example sum, so one backward per view gives the same gradient with less memory
            opt.zero_grad(set_to_none=True)
            loss = 0.0
            for x, t in views + [(eng, e_target)]:
                with torch.autocast("cuda", dtype=torch.float16):
                    part = (1 - (embed(model, x).float() * t).sum(1)).mean()
                scaler.scale(part).backward()
                loss += part.item()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            if step % args.log_every == 0:
                print(f"step {step}/{steps} loss {loss:.4f} lr {sched.get_last_lr()[0]:.2e} "
                      f"({(time.time() - t0) / step:.2f}s/step)", flush=True)
            if step % args.eval_every == 0 or step == steps:
                m = evaluate(model, dev, t_dev)
                print(f"eval step {step}: {m}", flush=True)
                if m["roman_to_native_r1"] > best and m["native_cos_to_teacher"] >= args.min_native_cos:
                    best = m["roman_to_native_r1"]
                    model.save(str(out_dir))
                    (out_dir / "train_state.json").write_text(json.dumps({"step": step, **m, **vars(args)}, indent=1))
                    print(f"  saved (best roman->native R@1 {best:.3f})", flush=True)
            if args.max_steps and step >= args.max_steps:
                return


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--precompute", action="store_true")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--max-steps", type=int, default=0, help="stop early (for a quick test)")
    p.add_argument("--min-native-cos", type=float, default=0.97, help="do not save if native embeddings drift more")
    p.add_argument("--mix", action="store_true", help="add the code-mixed view (needs data/train/lexicon.json)")
    p.add_argument("--mix-p", type=float, default=0.6, help="probability of swapping a lexicon word for English")
    p.add_argument("--out", default="models/indic-e5-small-phaseA")
    args = p.parse_args()
    precompute() if args.precompute else train(args)


if __name__ == "__main__":
    main()
