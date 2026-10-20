"""
EXP-001 local pilot: offline fuser arms (C, D1, D4) with a frozen prover, served by Ollama.

Runs stage by stage over all arms so each model is loaded once (OLLAMA_MAX_LOADED_MODELS=1):
pairs -> fuse -> formalize -> check -> prove -> verify -> summary.
Every stage appends raw records to JSONL before any filtering and skips records already on disk, so a run can be resumed.
Prompts, proof extraction and Lean checks reuse GAR's own code (PromptWriter, utils, LeanChecker); nothing in GAR is modified.
"""
import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time

import requests
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import PromptWriter, utils  # noqa: E402
from LeanChecker import LeanChecker  # noqa: E402
from filters import analysis_leak, glued  # noqa: E402

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"

# role -> (Ollama model, HF tokenizer used for GAR's chat template, sampling); settings follow GAR's effective defaults
MODELS = {
    "fuser": ("qwen3:8b", "Qwen/Qwen3-8B", dict(temperature=0.8, top_p=0.95, num_predict=16384)),
    "formalizer": ("hf.co/mradermacher/Goedel-Formalizer-V2-8B-GGUF:Q6_K", "Goedel-LM/Goedel-Formalizer-V2-8B",
                   dict(temperature=0.6, top_p=0.95, num_predict=8192)),
}
# Prover is chosen with --prover; both use GAR's prover sampling
PROVERS = {
    "goedel": ("hf.co/mradermacher/Goedel-Prover-V2-8B-GGUF:Q6_K", "Goedel-LM/Goedel-Prover-V2-8B",
               dict(temperature=1.0, top_p=0.95, num_predict=16385)),
    "deepseek": ("hf.co/unsloth/DeepSeek-Prover-V2-7B-GGUF:Q6_K", "deepseek-ai/DeepSeek-Prover-V2-7B",
                 dict(temperature=1.0, top_p=0.95, num_predict=16385)),
}
NUM_CTX = 17408  # prompt + longest generation; kept constant so Ollama never reloads a model mid-stage
# vLLM (GAR) samples with no top-k, no min-p and no repetition penalty; Ollama/llama.cpp defaults differ, so pin them
NEUTRAL_SAMPLING = dict(top_k=0, min_p=0.0, repeat_penalty=1.0)
# The prover runs on llama-server (see exp/README_local.md). The DeepSeek GGUFs lack the YaRN metadata (factor 16,
# orig ctx 4096) and Ollama cannot override it, so that server needs the extra rope flags below
PROVER_URL = "http://127.0.0.1:8081/completion"
PROVER_SERVER_ARGS = {"goedel": "env GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1; -c 17408", "deepseek": "--rope-scaling yarn --rope-scale 16 --yarn-orig-ctx 4096"}

LEAN_HINT = "The new statement must be formalizable in Lean 4 with Mathlib, using only standard, well-defined notions."

ARM_STEPS = {
    "D1": f"""1. List the skills each source problem requires.
2. Decide how to combine them so the new problem needs all of these skills together, not side by side. State how each source's key skill is used in the single combined problem.
3. Check that the new problem is solvable: its conditions are consistent and not contradictory.
4. Check that the new problem is verifiable: its answer is unique and checkable. {LEAN_HINT}""",
    "D4": f"""1. Inside <guideline></guideline>, write your own short guideline (about 100 tokens) for fusing these specific problems into a single harder problem that is still solvable and verifiable. {LEAN_HINT}
2. Follow your guideline step by step to draft the new problem.""",
}


def req_seed(*parts):
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest(), 16) % (2 ** 31)


def generate(role, prompt, seed):
    model, _, opts = MODELS[role]
    t0 = time.time()
    if role == "prover":
        r = requests.post(PROVER_URL, json={"prompt": prompt, "seed": seed, "cache_prompt": True, "n_predict": opts["num_predict"],
                                            "temperature": opts["temperature"], "top_p": opts["top_p"], **NEUTRAL_SAMPLING},
                          timeout=7200)
        r.raise_for_status()
        d = r.json()
        return {"text": d["content"], "gen_tokens": d["timings"]["predicted_n"], "prompt_tokens": d["tokens_evaluated"],
                "truncated": d.get("stop_type") == "limit", "seconds": round(time.time() - t0, 2)}
    r = requests.post(OLLAMA_URL, json={
        "model": model, "prompt": prompt, "raw": True, "stream": False, "keep_alive": "30m",
        "options": {**opts, **NEUTRAL_SAMPLING, "seed": seed, "num_ctx": NUM_CTX},
    }, timeout=7200)
    r.raise_for_status()
    d = r.json()
    return {"text": d["response"], "gen_tokens": d.get("eval_count", 0), "prompt_tokens": d.get("prompt_eval_count", 0),
            "truncated": d.get("done_reason") == "length", "seconds": round(time.time() - t0, 2)}


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path, rec):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def sample_pairs(base_path, n, seed):
    pool = load_jsonl(base_path)
    rng = random.Random(seed)
    pairs = []
    for i in range(n):
        a, b = rng.sample(pool, 2)  # random pairing (GAR default), k=2
        pairs.append({"pair_id": f"pair_{i:05d}", "sources": [a, b]})
    return pairs


def fuse_prompt(arm, sources, tok):
    prompt = PromptWriter.formulate_prompt_fuse_statements(
        [s["Natural_language_statement"] for s in sources], tok, skip_thinking=True)  # effective GAR baseline is reinit thinking
    if arm == "C":
        return prompt
    # Swap only the numbered <analysis> steps so the framing, sources and output format stay identical to arm C
    head, rest = prompt.split("enclosed within <analysis></analysis>.\n", 1)
    c_steps, tail = rest.split("\n\nAfter your analysis", 1)
    return f"{head}enclosed within <analysis></analysis>.\n{ARM_STEPS[arm]}\n\nAfter your analysis{tail}"


def parse_fused(text):
    # Same extraction order as NLStatementFuser._fuse_statement
    for code_type in ("md", "markdown", "Markdown"):
        if utils.contains_code_block(text, code_type=code_type):
            return utils.extract_code_blocks_as_list(text, code_type=code_type)[-1]
    if "<problem>" in text and "</problem>" in text:
        return text.split("<problem>")[-1].split("</problem>")[0].strip()
    return None


def parse_formalized(text):
    # Same as Autoformalizer._autoformalization (Goedel branch)
    blocks = utils.extract_code_blocks_as_list(text, code_type="lean4")
    if not blocks:
        return None
    m = re.search(r'theorem\b[\s\S]*?:=\s*by\b[\s\S]*', blocks[-1])
    return m.group(0) if m else None


def incomplete_proof(proof):
    """A lean4 block with no `:=` restates the theorem without proving it; it fails, but is not a statement edit."""
    return ":=" not in utils.remove_comments(proof)


def score_statement(lc, pid, statement, proofs, mod_check):
    """Return (pass_flags, n_modified, n_incomplete) for one statement's extracted proofs.
    gar: GAR's raw-text statement check inside LeanChecker.verify_proof (comments count as modifications).
    nocomment: Lean-check the proof alone; it passes only if the statement, with comments removed on both sides and
    whitespace ignored, still appears in the proof. Incomplete blocks (no `:=`) fail and are not counted as modified."""
    ids = [f"{pid}_{i}" for i in range(len(proofs))]
    if mod_check == "gar":
        res = lc.verify_proof(proofs, id_list=ids, original_statement=[statement] * len(proofs))
        modified = [utils.judge_statement_modification(statement, pr) for pr in proofs]
        return [x["pass"] for x in res], sum(modified), sum(map(incomplete_proof, proofs))
    res = lc.verify_proof(proofs, id_list=ids)
    incomplete = [incomplete_proof(pr) for pr in proofs]
    modified = [not inc and utils.judge_statement_modification(utils.remove_comments(statement), utils.remove_comments(pr))
                for pr, inc in zip(proofs, incomplete)]
    return [x["pass"] and not m and not inc for x, m, inc in zip(res, modified, incomplete)], sum(modified), sum(incomplete)


def summarize(fused, formal, checked, attempts, verified, n_pairs, filtered=()):
    """Per-arm metrics. 'useful' = valid statement and 0 < p <= 0.5. The *_clean metrics also exclude statements that
    fail the pre-prover filters (glued or analysis leak) and count only prover tokens spent on the statements that pass."""
    by_pair = lambda recs: {r["pair_id"]: r for r in recs}
    formal, checked, verified = by_pair(formal), by_pair(checked), by_pair(verified)
    bad = {f["pair_id"] for f in filtered if f["glued"] or f["analysis_leak"]}
    useful = [v for v in verified.values() if 0 < v["p_gar"] <= 0.5]
    useful_clean = [v for v in useful if v["pair_id"] not in bad]
    fuser_tok = sum(r["gen_tokens"] for r in fused)
    formal_tok = sum(r["gen_tokens"] for r in formal.values())
    prover_tok = sum(a["gen_tokens"] for a in attempts)
    total = fuser_tok + formal_tok + prover_tok
    total_clean = fuser_tok + formal_tok + sum(a["gen_tokens"] for a in attempts if a["pair_id"] not in bad)
    ps = [v["p_gar"] for v in verified.values()]
    return {
        "n_pairs": n_pairs,
        "fuse_parse_rate": sum(r["problem"] is not None for r in fused) / max(len(fused), 1),
        "formalize_rate": sum(r["statement"] is not None for r in formal.values()) / max(len(fused), 1),
        "validity_rate": sum(r["valid"] for r in checked.values()) / max(len(fused), 1),
        "n_verified": len(verified),
        "yield": len(useful) / max(len(fused), 1),
        # p_all counts truncated/unextracted attempts as failures (GAR's p_gar drops them from the denominator)
        "yield_p_all": sum(0 < v["p_all"] <= 0.5 for v in verified.values()) / max(len(fused), 1),
        "p_hist": {"p=0": sum(p == 0 for p in ps), "0<p<=0.5": sum(0 < p <= 0.5 for p in ps), "p>0.5": sum(p > 0.5 for p in ps)},
        "modification_rate": sum(v["m"] for v in verified.values()) / max(len(verified), 1),
        "incomplete_proof_rate": sum(v.get("incomplete", 0.0) for v in verified.values()) / max(len(verified), 1),
        "fuser_tokens_mean": fuser_tok / max(len(fused), 1),
        "fuser_truncation_rate": sum(r["truncated"] for r in fused) / max(len(fused), 1),
        "prover_tokens_mean": prover_tok / max(len(attempts), 1),
        "prover_truncation_rate": sum(a["truncated"] for a in attempts) / max(len(attempts), 1),
        "tokens_total": total,
        "tokens_per_useful": total / len(useful) if useful else None,
        "n_useful": len(useful),
        "n_filtered": len(filtered),
        "glued_rate": sum(f["glued"] for f in filtered) / max(len(filtered), 1),
        "analysis_leak_rate": sum(f["analysis_leak"] for f in filtered) / max(len(filtered), 1),
        "n_useful_clean": len(useful_clean),
        "yield_clean": len(useful_clean) / max(len(fused), 1),
        "tokens_total_clean": total_clean,
        "tokens_per_useful_clean": total_clean / len(useful_clean) if useful_clean else None,
    }


def selftest():
    assert parse_fused("x\n```md\nP1\n```\n```md\nP2\n```") == "P2"
    assert parse_fused("<problem>Q</problem>") == "Q" and parse_fused("none") is None
    assert parse_formalized("```lean4\ntheorem t : 1 = 1 := by\n  rfl\n```").startswith("theorem t")
    fused = [{"pair_id": "a", "problem": "x", "gen_tokens": 10, "truncated": False},
             {"pair_id": "b", "problem": None, "gen_tokens": 5, "truncated": True}]
    formal = [{"pair_id": "a", "statement": "theorem", "gen_tokens": 20}]
    checked = [{"pair_id": "a", "valid": True}]
    attempts = [{"pair_id": "a", "gen_tokens": 100, "truncated": False}] * 4
    verified = [{"pair_id": "a", "p_gar": 0.25, "p_all": 0.25, "m": 0.0}]
    s = summarize(fused, formal, checked, attempts, verified, 2)
    assert s["yield"] == 0.5 and s["tokens_per_useful"] == 435 and s["fuser_truncation_rate"] == 0.5
    assert s["n_useful_clean"] == 1 and s["tokens_per_useful_clean"] == 435  # no filter records: nothing excluded
    s = summarize(fused, formal, checked, attempts, verified, 2, [{"pair_id": "a", "glued": True, "analysis_leak": False}])
    assert s["n_useful"] == 1 and s["n_useful_clean"] == 0 and s["tokens_total_clean"] == 35 and s["glued_rate"] == 1.0
    assert incomplete_proof("theorem t (x : ℕ) :\n  x = x") and not incomplete_proof("theorem t : 1 = 1 := by\n  rfl")
    assert incomplete_proof("theorem t : P -- note := by") and not incomplete_proof("theorem t : 1 = 1 := by sorry")
    print("selftest ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["C", "D1", "D4"], choices=["C", "D1", "D4"])
    ap.add_argument("--n_pairs", type=int, default=20)
    ap.add_argument("--proofs", type=int, default=16)
    ap.add_argument("--prover", default="goedel", choices=list(PROVERS), help="goedel = GAR main.py default")
    ap.add_argument("--prover_max_tokens", type=int, default=16385, help="GAR uses 16385; lower only for local budget")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base_data", default="base_data/GAR_base_data_Numina-Math.json")
    ap.add_argument("--out", default="exp_results/EXP-001-local")
    ap.add_argument("--lean_url", default="http://localhost:80")
    ap.add_argument("--mod_check", default="gar", choices=["gar", "nocomment"],
                    help="gar = GAR's raw-text statement check; nocomment ignores comments (writes verify_nocomment.jsonl)")
    ap.add_argument("--prefilter", action="store_true", help="skip proving statements that are glued or leak <analysis>")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()

    sys.stdout.reconfigure(encoding="utf-8")
    MODELS["prover"] = PROVERS[args.prover]
    MODELS["prover"][2]["num_predict"] = args.prover_max_tokens
    os.makedirs(args.out, exist_ok=True)
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    ollama_v = subprocess.run(["ollama", "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-1]
    with open(f"{args.out}/config_{int(time.time())}.json", "w", encoding="utf-8") as f:
        json.dump({**vars(args), "git": git, "ollama": ollama_v, "models": MODELS, "num_ctx": NUM_CTX,
                   "neutral_sampling": NEUTRAL_SAMPLING, "prover_url": PROVER_URL, "prover_server": PROVER_SERVER_ARGS[args.prover],
                   "kv_cache": os.environ.get("OLLAMA_KV_CACHE_TYPE"), "arm_steps": ARM_STEPS}, f, indent=1)

    pairs_path = f"{args.out}/pairs_seed{args.seed}.jsonl"
    pairs = load_jsonl(pairs_path)
    if len(pairs) < args.n_pairs:  # extend deterministically; same seed gives the same prefix
        pairs = sample_pairs(args.base_data, args.n_pairs, args.seed)
        with open(pairs_path, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(p, ensure_ascii=False) + "\n" for p in pairs)
    pairs = pairs[:args.n_pairs]
    toks = {role: AutoTokenizer.from_pretrained(m[1]) for role, m in MODELS.items()}
    path = lambda arm, stage: f"{args.out}/{arm}/{stage}.jsonl"
    keep = {p["pair_id"] for p in pairs}  # downstream stages only see the first --n_pairs pairs
    load = lambda arm, stage: [r for r in load_jsonl(path(arm, stage)) if r["pair_id"] in keep]
    for arm in args.arms:
        os.makedirs(f"{args.out}/{arm}", exist_ok=True)

    def todo(arm, stage, items, key=lambda x: x["pair_id"]):
        done = {key(r) for r in load(arm, stage)}
        return [x for x in items if key(x) not in done]

    print("== fuse")
    for arm in args.arms:
        for p in todo(arm, "fuse", pairs):
            prompt = fuse_prompt(arm, p["sources"], toks["fuser"])
            g = generate("fuser", prompt, req_seed(args.seed, "fuse", arm, p["pair_id"]))
            append_jsonl(path(arm, "fuse"), {"pair_id": p["pair_id"], "arm": arm, "prompt": prompt, **g,
                                             "problem": parse_fused(g["text"])})
            print(f"[fuse {arm}] {p['pair_id']} {g['gen_tokens']} tok {g['seconds']}s")

    print("== formalize")
    for arm in args.arms:
        fused = [r for r in load(arm, "fuse") if r["problem"] is not None]
        for r in todo(arm, "formalize", fused):
            prompt = PromptWriter.formulate_prompt_autoformalization_Goedel("thm_GAR", r["problem"], toks["formalizer"])
            g = generate("formalizer", prompt, req_seed(args.seed, "formalize", arm, r["pair_id"]))
            append_jsonl(path(arm, "formalize"), {"pair_id": r["pair_id"], "problem": r["problem"], "prompt": prompt, **g,
                                                  "statement": parse_formalized(g["text"])})
            print(f"[formalize {arm}] {r['pair_id']} {g['gen_tokens']} tok {g['seconds']}s")

    print("== check")
    lc = LeanChecker(args.lean_url)
    for arm in args.arms:
        recs = [r for r in todo(arm, "check", load(arm, "formalize")) if r["statement"] is not None]
        if recs:
            for r, res in zip(recs, lc.verify_statement([r["statement"] for r in recs])):
                blocks = re.findall(r'(theorem.*?:= by)', r["statement"], re.DOTALL)
                append_jsonl(path(arm, "check"), {"pair_id": r["pair_id"], "NL": r["problem"], "valid": res["pass"],
                                                  "Statement": blocks[-1] if blocks else None, "messages": res["verify_result"]})

    print("== filter")
    for arm in args.arms:
        for r in todo(arm, "filter", [r for r in load(arm, "check") if r["valid"]]):
            g, parts, groups = glued(r["Statement"])
            append_jsonl(path(arm, "filter"), {"pair_id": r["pair_id"], "glued": g, "analysis_leak": analysis_leak(r["NL"]),
                                               "parts": parts, "groups": groups})

    print("== prove")
    for arm in args.arms:
        bad = {f["pair_id"] for f in load(arm, "filter") if f["glued"] or f["analysis_leak"]} if args.prefilter else set()
        valid = [r for r in load(arm, "check") if r["valid"] and r["pair_id"] not in bad]
        items = [dict(r, attempt=k) for r in valid for k in range(args.proofs)]
        for it in todo(arm, "prove", items, key=lambda x: (x["pair_id"], x["attempt"])):
            prompt = PromptWriter.formulate_prompt_proof_writing_Goedel_V2(it["NL"], it["Statement"], toks["prover"])
            # The GGUF has add_bos_token=true, so the server prepends BOS itself; drop the template's BOS to avoid a double BOS
            prompt = prompt.removeprefix(toks["prover"].bos_token or "")
            g = generate("prover", prompt, req_seed(args.seed, "prove", arm, it["pair_id"], it["attempt"]))
            blocks = utils.extract_code_blocks_as_list(g["text"], code_type="lean4")
            append_jsonl(path(arm, "prove"), {"pair_id": it["pair_id"], "attempt": it["attempt"], **g,
                                              "proof": blocks[-1] if blocks else None})
            print(f"[prove {arm}] {it['pair_id']}#{it['attempt']} {g['gen_tokens']} tok {g['seconds']}s trunc={g['truncated']}")

    print("== verify")
    # v2 of the nocomment scorer stops counting incomplete blocks as modifications; v1 files are kept as they were
    verify_stage = "verify" if args.mod_check == "gar" else f"verify_{args.mod_check}_v2"
    for arm in args.arms:
        statements = {r["pair_id"]: r["Statement"] for r in load(arm, "check") if r["valid"]}
        attempts = load(arm, "prove")
        for pid in sorted({a["pair_id"] for a in attempts} - {r["pair_id"] for r in load(arm, verify_stage)}):
            mine = [a for a in attempts if a["pair_id"] == pid]
            if len(mine) < args.proofs:
                continue  # proving for this statement is not finished yet
            proofs = [a["proof"] for a in mine if a["proof"] is not None]  # GAR drops attempts with no lean4 block
            flags, modified, incomplete = score_statement(lc, pid, statements[pid], proofs, args.mod_check) if proofs else ([], 0, 0)
            passes = sum(flags)
            append_jsonl(path(arm, verify_stage), {
                "pair_id": pid, "n_attempts": len(mine), "n_extracted": len(proofs), "passes": passes,
                "p_gar": passes / len(proofs) if proofs else 0.0,  # GAR's denominator: extracted proofs only
                "p_all": passes / len(mine), "m": modified / len(proofs) if proofs else 0.0,
                "incomplete": incomplete / len(proofs) if proofs else 0.0,
                "pass_flags": flags})

    print("== summary")
    summary = {arm: summarize(load(arm, "fuse"), load(arm, "formalize"), load(arm, "check"),
                              load(arm, "prove"), load(arm, verify_stage), args.n_pairs, load(arm, "filter"))
               for arm in args.arms}
    summary_name = "summary.json" if args.mod_check == "gar" else f"summary_{args.mod_check}_v2.json"
    with open(f"{args.out}/{summary_name}", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
