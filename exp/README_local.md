# Local EXP-001 pilot (AMD RX 6700 XT, Windows)

Only one model fits in VRAM at a time, so run the fuser side and the prover side separately.

1. Fuser side (Ollama: fuser + formalizer), with llama-server **stopped**:
   ```
   python -u exp/exp001_local.py --arms C D1 D4 --n_pairs 20 --proofs 0 --out exp_results/EXP-001-local-v2
   ```
2. Unload Ollama models (`ollama stop <model>`), then start the prover on llama-server (llama.cpp b11716, Vulkan).
   The YaRN flags are required: the GGUF lacks the rope-scaling metadata from the HF config.
   ```
   llama-server.exe -m <ollama blob of hf.co/unsloth/DeepSeek-Prover-V2-7B-GGUF:Q6_K> -ngl 99 -c 17408 -np 1 -fa on -ctk q8_0 -ctv q8_0 --rope-scaling yarn --rope-scale 16 --yarn-orig-ctx 4096 --host 127.0.0.1 --port 8081 --no-webui
   ```
   Find the blob with `ollama show --modelfile <model>` (the `FROM` line).
3. Prover side (same command, now with attempts; earlier stages are skipped because their JSONL exists):
   ```
   python -u exp/exp001_local.py --arms C D1 D4 --n_pairs 20 --proofs 4 --out exp_results/EXP-001-local-v2
   ```

`exp/prover_sanity.py` checks the prover on easy MiniF2F problems before trusting pass rates.
