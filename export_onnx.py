"""Export a model to ONNX (fp32 + dynamic int8) in the layout transformers.js expects, and check parity with
sentence-transformers. Run with the separate ONNX venv (optimum-onnx pins transformers 4.x):

    .venv-onnx/Scripts/python export_onnx.py models/release/indic-e5-small   (writes <folder>/onnx/)
"""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

src = Path(sys.argv[1])
onnx_dir = src / "onnx"
onnx_dir.mkdir(exist_ok=True)
if not (onnx_dir / "model.onnx").exists() or "--force" in sys.argv:
    with tempfile.TemporaryDirectory() as tmp:  # the bare transformer; pooling + normalize are done by the caller
        subprocess.run([sys.executable, "-m", "optimum.exporters.onnx", "--model", str(src), "--task",
                        "feature-extraction", "--library-name", "transformers", tmp], check=True)
        shutil.move(str(Path(tmp) / "model.onnx"), onnx_dir / "model.onnx")
if not (onnx_dir / "model_quantized.onnx").exists() or "--force" in sys.argv:
    quantize_dynamic(onnx_dir / "model.onnx", onnx_dir / "model_quantized.onnx", weight_type=QuantType.QInt8, per_channel=True)

dev = [json.loads(line) for line in open("data/train/dev.jsonl", encoding="utf-8")][::13]
texts = [p + t for r in dev for p, t in (("query: ", r["roman"]), ("passage: ", r["native"]))]
ref = SentenceTransformer(str(src), device="cpu").encode(texts, normalize_embeddings=True, batch_size=64)
tok = AutoTokenizer.from_pretrained(src)
for name in ["model.onnx", "model_quantized.onnx"]:
    sess = ort.InferenceSession(str(onnx_dir / name), providers=["CPUExecutionProvider"])
    embs = []
    for i in range(0, len(texts), 64):
        enc = tok(texts[i:i + 64], padding=True, truncation=True, max_length=512, return_tensors="np")
        names = {x.name for x in sess.get_inputs()}
        feed = {k: v for k, v in enc.items() if k in names}
        if "token_type_ids" in names and "token_type_ids" not in feed:  # as transformers.js does
            feed["token_type_ids"] = np.zeros_like(enc["input_ids"])
        h = sess.run(None, feed)[0]
        m = enc["attention_mask"][..., None]
        e = (h * m).sum(1) / m.sum(1)
        embs.append(e / np.linalg.norm(e, axis=1, keepdims=True))
    e = np.concatenate(embs)
    q, d = e[0::2], e[1::2]
    r1 = float(((q @ d.T).argmax(1) == np.arange(len(q))).mean())
    print(f"{name}: {(onnx_dir / name).stat().st_size / 1e6:.0f} MB, cosine to sentence-transformers "
          f"min {(e * ref).sum(1).min():.4f} mean {(e * ref).sum(1).mean():.4f}, dev roman->native R@1 {r1:.3f}")
rq, rd = ref[0::2], ref[1::2]
print(f"sentence-transformers dev roman->native R@1 {float(((rq @ rd.T).argmax(1) == np.arange(len(rq))).mean()):.3f}")
