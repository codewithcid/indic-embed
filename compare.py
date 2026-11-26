"""One table: MTEB(Indic, v1) (our runs, else published results) + romanized/code-mixed evaluation.

Only models with all 20 MTEB(Indic) tasks get an MTEB mean, so the means are comparable.
    python compare.py > results/baselines.md
"""

import glob
import json
import sys

import mteb
import pandas as pd

PARAMS_M = {
    "intfloat/multilingual-e5-small": 118, "intfloat/multilingual-e5-base": 278, "intfloat/multilingual-e5-large": 560,
    "BAAI/bge-m3": 568, "sentence-transformers/LaBSE": 471, "krutrim-ai-labs/Vyakyarth": 278,
    "l3cube-pune/indic-sentence-similarity-sbert": 238, "Qwen/Qwen3-Embedding-0.6B": 596,
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": 118,
    "ibm-granite/granite-embedding-107m-multilingual": 107, "Snowflake/snowflake-arctic-embed-m-v2.0": 305,
    "jinaai/jina-embeddings-v3": 572,
    # ours (run_mteb.py --like registers local models as local/<dir>)
    "local/e5-small-indic-48k": 40.5, "local/e5-small-indic-64k": 46.8, "local/indic-e5-small-phaseA": 46.8,
    "local/indic-e5-small-phaseA2": 46.8,
}


def main() -> None:
    bench = mteb.get_benchmark("MTEB(Indic, v1)")
    types = {t.metadata.name: t.metadata.type for t in bench.tasks}
    published = pd.read_csv("results/mteb_indic_published.csv").set_index("task_name")
    ours = mteb.ResultCache("results/mteb").load_results(models=list(PARAMS_M), tasks=bench).to_dataframe()
    ours = ours.set_index("task_name").drop(columns=["is_public"], errors="ignore")
    scores = published.reindex(columns=sorted(set(published.columns) | set(ours.columns)))
    for m in ours.columns:  # our runs fill gaps and take precedence
        scores[m] = ours[m].combine_first(scores[m]) if m in scores else ours[m]

    rows = []
    for m in scores.columns:
        s = scores[m]
        row = {"model": m, "params_M": PARAMS_M.get(m), "mteb_tasks": int(s.notna().sum()),
               "mteb_indic_mean": s.mean() if s.notna().sum() == len(types) else None}
        for ty in ("Retrieval", "BitextMining", "Classification", "STS"):
            row[ty] = s[[types.get(t) == ty for t in s.index]].mean() if s.notna().sum() == len(types) else None
        rows.append(row)
    table = pd.DataFrame(rows).set_index("model")

    for f in glob.glob("results/codemix/*.json"):
        d = json.load(open(f, encoding="utf-8"))
        s = d["summary"]
        table.loc[d["model"], "roman_to_native_R@1"] = s["dakshina_roman_to_native_r1"]
        table.loc[d["model"], "native_to_roman_R@1"] = s["dakshina_native_to_roman_r1"]
        table.loc[d["model"], "TanglishSTS"] = s["tanglish_sts_spearman"]

    table = table.sort_values("mteb_indic_mean", ascending=False)
    sys.stdout.reconfigure(encoding="utf-8")
    print(table.round(3).to_markdown())


if __name__ == "__main__":
    main()
