#!/usr/bin/env bash
# Waits for run_baselines.sh to finish, then fills bge-m3's last MTEB gaps and retries e5-large's code-mixed eval.
PY=.venv/Scripts/python
export PYTHONIOENCODING=utf-8
until grep -q "^=== done" results/baselines.log; do sleep 30; done
echo "=== $(date +%H:%M) bge-m3 remaining tasks"
$PY run_mteb.py BAAI/bge-m3 --batch-size 16 --tasks IN22ConvBitextMining BengaliSentimentAnalysis HindiDiscourseClassification MalayalamNewsClassification SanskritShlokasClassification UrduRomanSentimentClassification 2>&1 | grep -E "tasks in|^  [A-Z]|Error|out of memory"
echo "=== $(date +%H:%M) e5-large codemix (batch 8)"
$PY eval_codemix.py intfloat/multilingual-e5-large --max-per-lang 2000 --batch-size 8 2>&1 | grep -E "dakshina|tanglish|Error|out of memory|multilingual"
echo "=== done $(date +%H:%M)"
