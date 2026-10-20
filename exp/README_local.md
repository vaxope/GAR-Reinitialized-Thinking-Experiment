# Local EXP-001 pilot (AMD RX 6700 XT, Windows)

Only one model fits in VRAM at a time, so run the fuser side and the prover side separately.

1. Fuser side (Ollama: fuser + formalizer), with llama-server **stopped**:
   ```
   python -u exp/exp001_local.py --arms C D1 D4 --n_pairs 20 --proofs 0 --out exp_results/EXP-001-local-v2
   ```
2. Unload Ollama models (`ollama stop <model>`), then start the prover on llama-server (llama.cpp b11716, Vulkan).
   Set `GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1`. Without it, decode drops from ~43 tok/s to ~4 tok/s once the KV cache passes roughly 1 GB
   (the driver places it in host-visible memory). With it, a full 16,384-token generation runs at ~36.5 tok/s (448 s).
   ```
   set GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1
   llama-server.exe -m <ollama blob of hf.co/mradermacher/Goedel-Prover-V2-8B-GGUF:Q6_K> -ngl 99 -c 17408 -np 1 -fa on -ctk q8_0 -ctv q8_0 --host 127.0.0.1 --port 8081 --no-webui
   ```
   For `--prover deepseek` add `--rope-scaling yarn --rope-scale 16 --yarn-orig-ctx 4096` (the GGUF lacks the YaRN metadata) and use `-c 5120`.
   Find the blob with `ollama show --modelfile <model>` (the `FROM` line).
3. Prover side (same command, now with attempts; earlier stages are skipped because their JSONL exists):
   ```
   python -u exp/exp001_local.py --arms C D1 D4 --n_pairs 15 --proofs 4 --prover goedel --prover_max_tokens 8192 --out exp_results/EXP-001-local-v3-goedel
   ```

`exp/prover_sanity.py` checks the prover on easy MiniF2F problems before trusting pass rates.

Options: `--mod_check nocomment` ignores comments in the statement-modification check and treats lean4 blocks with no `:=` as incomplete, not modified (writes `verify_nocomment_v2.jsonl`) (default `gar` = GAR's raw-text check);
`--prefilter` skips proving glued statements and fused problems that leaked the `<analysis>` text (`exp/filters.py`).
To re-score an existing run without new GPU work, rerun the same command with `--mod_check nocomment`; finished stages are skipped.
