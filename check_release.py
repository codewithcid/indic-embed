"""Load a release folder in this environment and compare with the reference embeddings from prepare_release.py."""
import json
import sys
from pathlib import Path

import numpy as np
import sentence_transformers
import transformers
from sentence_transformers import SentenceTransformer

out = Path(sys.argv[1])
probe = json.loads((out.parent / (out.name + ".probe.json")).read_text(encoding="utf-8"))
ref = np.load(out.parent / (out.name + ".probe.npy"))
model = SentenceTransformer(str(out), device="cpu")
e = model.encode(probe, normalize_embeddings=True)
print(f"sentence-transformers {sentence_transformers.__version__}, transformers {transformers.__version__}: "
      f"{type(model.tokenizer).__name__}, cosine to reference min {float((e * ref).sum(1).min()):.6f}")
