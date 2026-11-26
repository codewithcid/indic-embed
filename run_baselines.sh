#!/usr/bin/env bash
# Step 1 baselines: MTEB(Indic) gaps + code-mixed eval for every candidate. Sequential (one 6 GB GPU).
set -u
PY=.venv/Scripts/python
export PYTHONIOENCODING=utf-8
run() { echo "=== $(date +%H:%M) $*"; "$@" 2>&1 | grep -E "tasks in|^  [A-Z]|dakshina|tanglish|Error|error:" ; }
run $PY run_mteb.py BAAI/bge-m3 --batch-size 32 --tasks BelebeleRetrieval XQuADRetrieval WikipediaRerankingMultilingual SIB200ClusteringS2S MTOPIntentClassification TweetSentimentClassification IndicCrosslingualSTS XNLI MultiHateClassification
run $PY run_mteb.py intfloat/multilingual-e5-large --batch-size 32
run $PY run_mteb.py krutrim-ai-labs/Vyakyarth --batch-size 32
run $PY run_mteb.py l3cube-pune/indic-sentence-similarity-sbert --batch-size 32
for m in intfloat/multilingual-e5-base intfloat/multilingual-e5-large BAAI/bge-m3 sentence-transformers/LaBSE krutrim-ai-labs/Vyakyarth l3cube-pune/indic-sentence-similarity-sbert sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 ibm-granite/granite-embedding-107m-multilingual; do
  run $PY eval_codemix.py $m --max-per-lang 2000 --batch-size 32
done
echo "=== done $(date +%H:%M)"
