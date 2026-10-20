#!/usr/bin/env bash
# EXP-003 fuser side (no prover): tagging for both domains, puzzle formalization faithfulness, then
# fuse -> formalize -> Lean check -> filters for six cells per domain. Every step is resumable and skips finished work.
# Usage: bash exp/run_exp003_fuserside.sh   (from the repo root; Ollama running, llama-server stopped)
set -euo pipefail
CELLS="C:random D1:random D2:random D3:random D2:same_tag D2:band"
# Pairing thresholds are calibrated per domain on pairwise cosine percentiles (the fixed 0.5-0.8 / 0.9 numbers in
# CLAUDE.md assume a different embedding scale): near-duplicate cutoff = 99th percentile, band = [90th, 99th).
q() { python -c "import json;print(json.load(open('tagging/$1/validation_summary.json'))['pairwise_cosine_quantiles']['$2'])"; }

python -u exp/tagging.py label propagate validate slots --domain math --n_label 5000
python -u exp/tagging.py embed cluster name label propagate validate slots --domain puzzles \
  --desc "logic and constraint puzzles" --meta_key family --k 16 --max_topics 10 --n_label 1500
python -u exp/puzzle_faithfulness.py --per_family 6 --timeout 30

for D in math puzzles; do
  N=100; [ "$D" = puzzles ] && N=50
  python -u exp/exp001_local.py --domain $D --pool tagging/$D/pool.jsonl --cells $CELLS --n_pairs $N --proofs 0 \
    --band $(q $D 0.9) $(q $D 0.99) --dup_cut $(q $D 0.99) --out exp_results/EXP-003-$D
done
