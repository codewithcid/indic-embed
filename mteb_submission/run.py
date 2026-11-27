"""Run MTEB(Indic, v1) for the published model with its submission ModelMeta; results in mteb_submission/results."""
import sys
from pathlib import Path

import mteb

sys.path.insert(0, str(Path(__file__).parent))
from indic_e5_models import indic_e5_small as meta  # noqa: E402

model = meta.load_model() if hasattr(meta, "load_model") else meta.loader(meta.name, revision=meta.revision, **meta.loader_kwargs)
print(type(model).__name__, getattr(model, "mteb_model_meta", meta).name, getattr(model, "mteb_model_meta", meta).revision)
tasks = mteb.get_benchmark("MTEB(Indic, v1)").tasks
res = mteb.evaluate(model, tasks, encode_kwargs={"batch_size": 64}, cache=mteb.ResultCache("mteb_submission/cache"))
for r in res.task_results:
    print(f"  {r.task_name:40} {r.get_score():.4f}")
