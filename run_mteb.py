"""Run MTEB(Indic, v1) (or chosen tasks) for one model and save results under results/mteb/.

    python run_mteb.py intfloat/multilingual-e5-small
    python run_mteb.py BAAI/bge-m3 --tasks WikipediaRerankingMultilingual XQuADRetrieval
"""

import argparse
import time

import mteb


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--tasks", nargs="*", help="task names (default: all of MTEB(Indic, v1))")
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()

    bench = mteb.get_benchmark("MTEB(Indic, v1)")
    tasks = [t for t in bench.tasks if not args.tasks or t.metadata.name in args.tasks]
    model = mteb.get_model(args.model)
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
