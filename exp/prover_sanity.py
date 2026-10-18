"""Prover sanity check on easy MiniF2F problems through llama-server (checks the YaRN fix before trusting pass rates)."""
import argparse
import json
import os
import sys

import requests
import tokenizers
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import PromptWriter, utils  # noqa: E402
from LeanChecker import LeanChecker  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8081/completion")
ap.add_argument("--n", type=int, default=5)
ap.add_argument("--attempts", type=int, default=2)
ap.add_argument("--max_tokens", type=int, default=8192)
ap.add_argument("--out", default="exp_results/prover_sanity.jsonl")
args = ap.parse_args()
sys.stdout.reconfigure(encoding="utf-8")

tok = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-Prover-V2-7B")  # chat template only; its encode is broken in transformers 5.16
raw = tokenizers.Tokenizer.from_file(hf_hub_download("deepseek-ai/DeepSeek-Prover-V2-7B", "tokenizer.json"))
lc = LeanChecker()
data = json.load(open("data/test_dataset/MiniF2F_test_Lean4.json", encoding="utf-8"))[:args.n]
passed = total = 0
for x in data:
    stmt = x["Statement"] + " by"
    prompt = PromptWriter.formulate_prompt_proof_writing_Goedel_V2(x["Informal_statement"], stmt, tok).removeprefix(tok.bos_token)
    for s in range(args.attempts):
        d = requests.post(args.url, json={"prompt": prompt, "n_predict": args.max_tokens, "temperature": 1.0, "top_p": 0.95,
                                          "top_k": 0, "min_p": 0.0, "repeat_penalty": 1.0, "seed": s, "cache_prompt": True},
                          timeout=7200).json()
        blocks = utils.extract_code_blocks_as_list(d["content"], "lean4")
        ok = bool(blocks) and lc.verify_proof([blocks[-1]], original_statement=[stmt])[0]["pass"]
        passed += ok; total += 1
        t = d["timings"]
        rec = {"name": x["Name"], "seed": s, "prompt_tokens": d["tokens_evaluated"],
               "raw_tokenizer_prompt_tokens": len(raw.encode(prompt, add_special_tokens=False).ids),
               "gen_tokens": t["predicted_n"], "tok_s": round(t["predicted_per_second"], 1),
               "truncated": d.get("stop_type") == "limit", "extracted": bool(blocks), "pass": ok, "text": d["content"]}
        with open(args.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print({k: v for k, v in rec.items() if k != "text"}, flush=True)
print(f"passed {passed}/{total}")
