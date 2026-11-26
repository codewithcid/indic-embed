"""Run MTEB(Indic, v1) (or chosen tasks) for one model and save results under results/mteb/.

    python run_mteb.py intfloat/multilingual-e5-small
    python run_mteb.py BAAI/bge-m3 --tasks WikipediaRerankingMultilingual XQuADRetrieval
    python run_mteb.py models/e5-small-indic-64k --like intfloat/multilingual-e5-small   # local model, same prompts
"""

import argparse
import time
from pathlib import Path

import mteb


def load(model: str, like: str | None):
    if not like:
        return mteb.get_model(model)
    # A local model gets the same wrapper and prompts as `like`, under its own name.
    base = mteb.get_model_meta(like)
    name = f"local/{Path(model).name}"
    meta = base.model_copy(update={"name": name, "revision": "local", "n_parameters": None, "memory_usage_mb": None})
    wrapper = base.loader(model, **(base.loader_kwargs or {}))
    wrapper.mteb_model_meta = meta
    return wrapper


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--like", help="for a local model: the Hub model whose MTEB wrapper and prompts to reuse")
    p.add_argument("--tasks", nargs="*", help="task names (default: all of MTEB(Indic, v1))")
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()

    bench = mteb.get_benchmark("MTEB(Indic, v1)")
    tasks = [t for t in bench.tasks if not args.tasks or t.metadata.name in args.tasks]
    model = load(args.model, args.like)
    t0 = time.time()
    results = mteb.evaluate(
        model,
        tasks,
        encode_kwargs={"batch_size": args.batch_size},
        cache=mteb.ResultCache("results/mteb"),
        overwrite_strategy="only-missing",
    )
    print(f"{args.model}: {len(tasks)} tasks in {time.time() - t0:.0f}s")
    for r in results.task_results if hasattr(results, "task_results") else results:
        print(f"  {r.task_name:40} {r.get_score():.4f}")


if __name__ == "__main__":
    main()
