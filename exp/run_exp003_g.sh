#!/usr/bin/env bash
# EXP-003 option G: arms C (GAR), D1 (domain-agnostic skeleton), D2 (skeleton + topic checks) on random pairs, in
# three domains (math, puzzles, stem), all verified by Lean. Every step is resumable and skips finished work.
# 1) STEM tagging + fuser side (Ollama)  2) prover passes (llama-server, Goedel-Prover-V2-8B, 16k, 4 attempts).
# Usage: bash exp/run_exp003_g.sh   (from the repo root)
set -euo pipefail
CELLS="C:random D1:random D2:random"
LLAMA=/c/Users/stanh/tools/llama.cpp-b11716/llama-server.exe
GOEDEL=C:/Users/stanh/.ollama/models/blobs/sha256-43a23c094fd495304726f1471a9e10ca48997573278e306a555be1d3371aff3a
q() { python -c "import json;print(json.load(open('tagging/$1/validation_summary.json'))['pairwise_cosine_quantiles']['$2'])"; }
run() {  # domain n_pairs proofs [extra args]
  local D=$1 N=$2 P=$3; shift 3
  python -u exp/exp001_local.py --domain $D --pool tagging/$D/pool.jsonl --cells $CELLS --n_pairs $N --proofs $P \
    --band $(q $D 0.9) $(q $D 0.99) --dup_cut $(q $D 0.99) --out exp_results/EXP-003-$D "$@"
}

# 1) STEM: tagging and fuser side
python -u exp/tagging.py embed cluster name label propagate validate slots --domain stem \
  --desc "science questions (physics, chemistry, biology, engineering) with quantitative answers" \
  --meta_key category --k 20 --max_topics 12 --n_label 1500
run stem 40 0

# 2) prover: unload Ollama models, start llama-server with the long-context fix
for m in $(ollama ps 2>/dev/null | awk 'NR>1 {print $1}'); do ollama stop "$m"; done
GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1 "$LLAMA" -m "$GOEDEL" -ngl 99 -c 17408 -np 1 -fa on -ctk q8_0 -ctv q8_0 \
  --host 127.0.0.1 --port 8081 --no-webui > exp_results/llama_server_exp003.log 2>&1 &
until curl -s 127.0.0.1:8081/health | grep -q ok; do sleep 3; done

# pass 1 gives every domain x arm some statements early; pass 2 extends the same cells (prefix of the same pairs)
for pass in "14 25 20" "28 50 40"; do
  set -- $pass
  run math $1 4 --prefilter
  run puzzles $2 4 --prefilter
  run stem $3 4 --prefilter
done
