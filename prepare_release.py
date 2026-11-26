"""Copy a trained model into a release folder that loads on old and new sentence-transformers/transformers:
classic module paths and pooling config (as in intfloat/multilingual-e5-small), e5 prompts registered, model card.
Saves reference embeddings so other environments can check they get the same vectors (check_release.py).

    python prepare_release.py models/indic-e5-small-phaseA2 models/release/indic-e5-small
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import sentence_transformers
import torch
import transformers
from sentence_transformers import SentenceTransformer

src, out = Path(sys.argv[1]), Path(sys.argv[2])
if out.exists():
    shutil.rmtree(out)
(out / "1_Pooling").mkdir(parents=True)
for f in ["model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json"]:
    shutil.copy(src / f, out / f)
for f in ["special_tokens_map.json"]:
    if (src / f).exists():
        shutil.copy(src / f, out / f)
modules = [{"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"},
           {"idx": 1, "name": "1", "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
           {"idx": 2, "name": "2", "path": "2_Normalize", "type": "sentence_transformers.models.Normalize"}]
(out / "modules.json").write_text(json.dumps(modules, indent=2))
(out / "1_Pooling" / "config.json").write_text(json.dumps(
    {"word_embedding_dimension": 384, "pooling_mode_cls_token": False, "pooling_mode_mean_tokens": True,
     "pooling_mode_max_tokens": False, "pooling_mode_mean_sqrt_len_tokens": False}, indent=2))
(out / "sentence_bert_config.json").write_text(json.dumps({"max_seq_length": 512, "do_lower_case": False}, indent=2))
(out / "config_sentence_transformers.json").write_text(json.dumps(
    {"__version__": {"sentence_transformers": sentence_transformers.__version__, "transformers": transformers.__version__,
                     "pytorch": torch.__version__},
     "prompts": {"query": "query: ", "document": "passage: "}, "default_prompt_name": None,
     "similarity_fn_name": "cosine"}, indent=2))
shutil.copy("MODEL_CARD.md", out / "README.md")

probe = ["query: mujhe kal office jaana hai", "passage: मुझे कल ऑफिस जाना है", "query: naan nalaikku varuven",
         "passage: நான் நாளைக்கு வருவேன்", "query: How are you?", "passage: ঢাকা বাংলাদেশের রাজধানী"]
a = SentenceTransformer(str(src), device="cpu").encode(probe, normalize_embeddings=True)
b = SentenceTransformer(str(out), device="cpu").encode(probe, normalize_embeddings=True)
print("release vs source cosine min:", float((a * b).sum(1).min()))
q = SentenceTransformer(str(out), device="cpu").encode_query(["mujhe kal office jaana hai"])
print("encode_query adds the prefix:", float(q[0] @ b[0]) > 0.9999)
np.save(out.parent / (out.name + ".probe.npy"), b)
(out.parent / (out.name + ".probe.json")).write_text(json.dumps(probe, ensure_ascii=False), encoding="utf-8")
