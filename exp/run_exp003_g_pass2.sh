#!/usr/bin/env bash
# EXP-003 option G, pass 2 with the order changed on 2026-10-21 (user request): math, then stem, then puzzles.
# Puzzles were moved last because pass 1 solved almost none of them. Reuses the llama-server started by run_exp003_g.sh.
set -euo pipefail
CELLS="C:random D1:random D2:random"
q() { python -c "import json;print(json.load(open('tagging/$1/validation_summary.json'))['pairwise_cosine_quantiles']['$2'])"; }
run() {
  local D=$1 N=$2
  python -u exp/exp001_local.py --domain $D --pool tagging/$D/pool.jsonl --cells $CELLS --n_pairs $N --proofs 4 \
    --band $(q $D 0.9) $(q $D 0.99) --dup_cut $(q $D 0.99) --out exp_results/EXP-003-$D --prefilter
}
until curl -s 127.0.0.1:8081/health | grep -q ok; do sleep 3; done
run math 28
run stem 40
run puzzles 50
