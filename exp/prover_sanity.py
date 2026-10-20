"""
Local prover on MiniF2F-Test through llama-server, scored with GAR's default check (statement must appear verbatim;
sorry/admit fail). Used to sanity-check a prover setup and to calibrate the local quantized prover against published numbers.
Resumable: attempts already in --out are skipped.
"""
import argparse
import json
import os
import random
import sys
import time

import requests
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import PromptWriter, utils  # noqa: E402
from LeanChecker import LeanChecker  # noqa: E402
from exp001_local import NEUTRAL_SAMPLING, PROVERS, PROVER_URL, load_jsonl, append_jsonl  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--prover", default="goedel", choices=list(PROVERS))
ap.add_argument("--data", default="data/test_dataset/MiniF2F_test_Lean4.json")
ap.add_argument("--sample", type=int, default=5, help="number of problems; a seeded random subset unless --first")
ap.add_argument("--first", action="store_true", help="take the first --sample problems instead of a random subset")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--attempts", type=int, default=2)
ap.add_argument("--max_tokens", type=int, default=16385)
ap.add_argument("--out", default="exp_results/prover_sanity.jsonl")
args = ap.parse_args()
sys.stdout.reconfigure(encoding="utf-8")

_, tok_id, opts = PROVERS[args.prover]
tok = AutoTokenizer.from_pretrained(tok_id)  # chat template only (DeepSeek's encode is broken in transformers 5.16)
lc = LeanChecker()
data = json.load(open(args.data, encoding="utf-8"))
data = data[:args.sample] if args.first else random.Random(args.seed).sample(data, args.sample)
done = {(r["name"], r["seed"]) for r in load_jsonl(args.out)}
os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
with open(args.out.replace(".jsonl", "_config.json"), "w", encoding="utf-8") as f:
    json.dump({**vars(args), "model": PROVERS[args.prover], "problems": [x["Name"] for x in data], "time": int(time.time())}, f, indent=1)

for x in data:
    prompt = PromptWriter.formulate_prompt_proof_writing_Goedel_V2(x["Informal_statement"], x["Statement"], tok)
    prompt = prompt.removeprefix(tok.bos_token or "")  # the server adds BOS itself when the GGUF asks for it
    for s in range(args.attempts):
        if (x["Name"], s) in done:
            continue
        d = requests.post(PROVER_URL, json={"prompt": prompt, "n_predict": args.max_tokens, "temperature": opts["temperature"],
                                            "top_p": opts["top_p"], **NEUTRAL_SAMPLING, "seed": s, "cache_prompt": True},
                          timeout=7200).json()
        blocks = utils.extract_code_blocks_as_list(d["content"], "lean4")
        ok = bool(blocks) and lc.verify_proof([blocks[-1]], original_statement=[x["Statement"]])[0]["pass"]
        t = d["timings"]
        rec = {"name": x["Name"], "seed": s, "prompt_tokens": d["tokens_evaluated"], "gen_tokens": t["predicted_n"],
               "tok_s": round(t["predicted_per_second"], 1), "truncated": d.get("stop_type") == "limit",
               "extracted": bool(blocks), "pass": ok, "text": d["content"]}
        append_jsonl(args.out, rec)
        print({k: v for k, v in rec.items() if k != "text"}, flush=True)

recs = [r for r in load_jsonl(args.out) if r["name"] in {x["Name"] for x in data}]
by = {}
for r in recs:
    by.setdefault(r["name"], []).append(r["pass"])
print(f"problems {len(by)}, attempts {len(recs)}, pass@1 {sum(map(sum, by.values())) / max(len(recs), 1):.3f}, "
      f"solved (pass@{args.attempts}) {sum(any(v) for v in by.values())}/{len(by)}")
