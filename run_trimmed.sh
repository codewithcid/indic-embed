#!/usr/bin/env bash
# Benchmark trimmed models: MTEB(Indic) + romanized/code-mixed, with e5-small's prompts.
PY=.venv/Scripts/python
export PYTHONIOENCODING=utf-8
for m in "$@"; do
  echo "=== $(date +%H:%M) $m"
  $PY run_mteb.py models/$m --like intfloat/multilingual-e5-small --batch-size 64 2>&1 | grep -E "tasks in|Error|out of memory"
  $PY eval_codemix.py models/$m --like intfloat/multilingual-e5-small --max-per-lang 2000 --batch-size 64 2>&1 | grep -E "^local|Error|out of memory"
done
echo "=== done $(date +%H:%M)"
