"""
Local offline fuser experiments (EXP-001, EXP-003): fuser arms x source pairing, frozen prover, served by Ollama/llama.cpp.

Arms (only the numbered <analysis> steps differ): C = GAR, D1 = generic skeleton, D2 = skeleton + topic slots for the
sources' tags, D3 = skeleton + slots for wrong topics (control), D4 = self-written guideline.
Pairings (with --pool): random, same_tag (partner shares the anchor's confident top tag), band (embedding cosine in --band).
All pairings share the same anchors. A cell is ARM:PAIRING; its outputs go to <out>/<ARM>_<PAIRING>/.

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

import numpy as np
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

# Domain adapters: the hint appended to the skeleton, and whether brute-force tactics count as failures
DOMAIN_HINTS = {
    "math": LEAN_HINT,
    "puzzles": "The new puzzle must be formalizable in Lean 4 as a statement about finite objects, with its unique answer stated explicitly.",
    "stem": "The new problem must be formalizable in Lean 4 with Mathlib as a precise quantitative statement: give every quantity a value, state the model or law used, and state the unique answer.",
}
# Puzzles live on finite domains, where `decide` can check every case without any reasoning
FORBID_BRUTE_FORCE = {"math": False, "puzzles": True, "stem": False}
BRUTE_FORCE = re.compile(r"\b(native_decide|decide)\b")


def arm_steps(arm, hint=LEAN_HINT, topic_lines=()):
    if arm == "D4":
        return f"""1. Inside <guideline></guideline>, write your own short guideline (about 100 tokens) for fusing these specific problems into a single harder problem that is still solvable and verifiable. {hint}
2. Follow your guideline step by step to draft the new problem."""
    steps = f"""1. List the skills each source problem requires.
2. Decide how to combine them so the new problem needs all of these skills together, not side by side. State how each source's key skill is used in the single combined problem.
3. Check that the new problem is solvable: its conditions are consistent and not contradictory.
4. Check that the new problem is verifiable: its answer is unique and checkable. {hint}"""
    if topic_lines:  # D2 / D3
        steps += "\n5. Topic checks:\n" + "\n".join(f"- {t}: {slot}" for t, slot in topic_lines)
    return steps


ARM_STEPS = {a: arm_steps(a) for a in ("D1", "D4")}  # math versions, as used in EXP-001


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


def src_text(s):
    return s.get("text") or s["Natural_language_statement"]


def make_pairs(pool, n, seed, pairing, tags, E, conf, band, dup_cut=0.9):
    """Shared anchors for every pairing; the partner depends on the pairing. Partners at cosine >= dup_cut to the anchor
    (near-duplicates) are never used. A pairing with no candidate falls back to random and records why."""
    anchors = random.Random(seed).sample(range(len(pool)), n)
    top = [tags[r["id"]]["tags"][0] for r in pool]
    pairs = []
    for i, a in enumerate(anchors):
        rng = random.Random(req_seed(seed, pairing, i))
        sims = E[a] @ E.T
        ok = sims < dup_cut
        ok[a] = False
        cand, fallback = None, None
        if pairing == "same_tag":
            t, p = top[a]
            if p >= conf:
                cand = [j for j in np.flatnonzero(ok) if top[j][0] == t and top[j][1] >= conf]
            fallback = None if cand else ("anchor not confidently tagged" if p < conf else "no same-tag partner")
        elif pairing == "band":
            cand = list(np.flatnonzero(ok & (sims >= band[0]) & (sims <= band[1])))
            fallback = None if cand else "no partner in band"
        if not cand:
            cand = list(np.flatnonzero(ok))
        b = int(rng.choice(cand))
        pairs.append({"pair_id": f"pair_{i:05d}", "pairing": pairing, "sources": [pool[a], pool[b]],
                      "cos": round(float(sims[b]), 4), "fallback": fallback})
    return pairs


def topic_lines(arm, pair, tags, slots, conf, seed):
    """Slots for D2 (each source's own confident top tag) or D3 (a random wrong topic per confidently tagged source).
    Sources below the confidence threshold get no slot (generic skeleton). The slot candidate rotates with the pair
    index, so each candidate's yield can be compared later (the exploration phase of the slot bandit)."""
    if arm not in ("D2", "D3") or not tags:
        return []
    cand = int(pair["pair_id"].split("_")[-1]) % 3
    names = sorted(slots)
    out = []
    for k, s in enumerate(pair["sources"]):
        (t1, p1), (t2, _) = tags[s["id"]]["tags"][:2]
        if p1 < conf:
            continue
        t = t1 if arm == "D2" else random.Random(req_seed(seed, "D3", pair["pair_id"], k)).choice(
            [n for n in names if n not in (t1, t2)])
        if t not in [x[0] for x in out]:
            out.append((t, slots[t][cand]))
    return out


def fuse_prompt(arm, sources, tok, hint=LEAN_HINT, lines=()):
    prompt = PromptWriter.formulate_prompt_fuse_statements(
        [src_text(s) for s in sources], tok, skip_thinking=True)  # effective GAR baseline is reinit thinking
    if arm == "C":
        return prompt
    # Swap only the numbered <analysis> steps so the framing, sources and output format stay identical to arm C
    head, rest = prompt.split("enclosed within <analysis></analysis>.\n", 1)
    c_steps, tail = rest.split("\n\nAfter your analysis", 1)
    return f"{head}enclosed within <analysis></analysis>.\n{arm_steps(arm, hint, lines)}\n\nAfter your analysis{tail}"


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


def score_statement(lc, pid, statement, proofs, mod_check, forbid_brute_force=False):
    """Return (pass_flags, n_modified, n_incomplete, n_brute_force) for one statement's extracted proofs.
    With forbid_brute_force (puzzle domain) proofs using decide / native_decide fail; brute force is always counted.
    gar: GAR's raw-text statement check inside LeanChecker.verify_proof (comments count as modifications).
    nocomment: Lean-check the proof alone; it passes only if the statement, with comments removed on both sides and
    whitespace ignored, still appears in the proof. Incomplete blocks (no `:=`) fail and are not counted as modified."""
    ids = [f"{pid}_{i}" for i in range(len(proofs))]
    brute = [bool(BRUTE_FORCE.search(utils.remove_comments(pr))) for pr in proofs]
    ok_tactics = [not (forbid_brute_force and b) for b in brute]
    if mod_check == "gar":
        res = lc.verify_proof(proofs, id_list=ids, original_statement=[statement] * len(proofs))
        modified = [utils.judge_statement_modification(statement, pr) for pr in proofs]
        return ([x["pass"] and t for x, t in zip(res, ok_tactics)], sum(modified), sum(map(incomplete_proof, proofs)),
                sum(brute))
    res = lc.verify_proof(proofs, id_list=ids)
    incomplete = [incomplete_proof(pr) for pr in proofs]
    modified = [not inc and utils.judge_statement_modification(utils.remove_comments(statement), utils.remove_comments(pr))
                for pr, inc in zip(proofs, incomplete)]
    return ([x["pass"] and not m and not inc and t for x, m, inc, t in zip(res, modified, incomplete, ok_tactics)],
            sum(modified), sum(incomplete), sum(brute))


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
        "brute_force_rate": sum(v.get("brute_force", 0.0) for v in verified.values()) / max(len(verified), 1),
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
    # pairing + slots on a tiny synthetic pool: 4 problems, 2 topics, orthogonal-ish embeddings
    pool = [{"id": f"p{i}", "text": f"problem {i}"} for i in range(4)]
    tg = {"p0": {"tags": [["T1", 0.9], ["T2", 0.1]]}, "p1": {"tags": [["T1", 0.8], ["T2", 0.2]]},
          "p2": {"tags": [["T2", 0.9], ["T1", 0.1]]}, "p3": {"tags": [["T2", 0.4], ["T1", 0.3]]}}
    Em = np.array([[1, 0, 0], [0.6, 0.8, 0], [0, 1, 0], [0.7, 0, 0.714]], dtype=np.float32)
    for pr in make_pairs(pool, 4, 0, "same_tag", tg, Em, 0.5, (0.5, 0.8)):
        a, b = pr["sources"]
        assert pr["fallback"] or tg[a["id"]]["tags"][0][0] == tg[b["id"]]["tags"][0][0]
    for pr in make_pairs(pool, 4, 0, "band", tg, Em, 0.5, (0.5, 0.8)):
        assert pr["fallback"] or 0.5 <= pr["cos"] <= 0.8
    sl = {"T1": ["s1a", "s1b", "s1c"], "T2": ["s2a", "s2b", "s2c"], "T3": ["s3a", "s3b", "s3c"]}
    pr = {"pair_id": "pair_00001", "sources": [pool[0], pool[3]]}
    assert topic_lines("D2", pr, tg, sl, 0.5, 0) == [("T1", "s1b")]  # p3 is below the confidence threshold
    assert topic_lines("D3", pr, tg, sl, 0.5, 0) == [("T3", "s3b")]  # the only topic that is not p0's
    assert topic_lines("C", pr, tg, sl, 0.5, 0) == [] and arm_steps("D2", LEAN_HINT, []) == ARM_STEPS["D1"]
    assert incomplete_proof("theorem t (x : ℕ) :\n  x = x") and not incomplete_proof("theorem t : 1 = 1 := by\n  rfl")
    assert incomplete_proof("theorem t : P -- note := by") and not incomplete_proof("theorem t : 1 = 1 := by sorry")
    print("selftest ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["C", "D1", "D4"], choices=["C", "D1", "D2", "D3", "D4"])
    ap.add_argument("--cells", nargs="+", default=None,
                    help="ARM:PAIRING cells, e.g. C:random D2:same_tag D2:band (needs --pool); overrides --arms")
    ap.add_argument("--pool", default=None, help="source pool from exp/build_pools.py; default = GAR base data, random pairs")
    ap.add_argument("--domain", default="math", choices=list(DOMAIN_HINTS))
    ap.add_argument("--tags_dir", default=None, help="tagging output (tags.jsonl, slots.json); default tagging/<domain>")
    ap.add_argument("--tag_conf", type=float, default=0.5, help="tags below this confidence are ignored")
    ap.add_argument("--band", type=float, nargs=2, default=[0.5, 0.8], help="cosine band for the band pairing")
    ap.add_argument("--dup_cut", type=float, default=0.9, help="partners at or above this cosine count as near-duplicates")
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
                   "kv_cache": os.environ.get("OLLAMA_KV_CACHE_TYPE"), "arm_steps": ARM_STEPS,
                   "domain_hint": DOMAIN_HINTS[args.domain]}, f, indent=1)

    # cells: (arm, output dir, pairing); without --cells the EXP-001 layout is kept (dir = arm, random pairs)
    if args.cells:
        assert args.pool, "--cells needs --pool"
        cells = [(c.split(":")[0], c.replace(":", "_"), c.split(":")[1]) for c in args.cells]
    else:
        cells = [(a, a, "random") for a in args.arms]
    hint = DOMAIN_HINTS[args.domain]
    tags = slots = E = pool = None
    if args.pool:
        tags_dir = args.tags_dir or f"tagging/{args.domain}"
        pool = load_jsonl(args.pool)
        tags = {r["id"]: r for r in load_jsonl(f"{tags_dir}/tags.jsonl")}
        slots = json.load(open(f"{tags_dir}/slots.json", encoding="utf-8"))
        cache = os.path.join(os.path.expanduser("~"), "GAR-cache", args.domain)
        E = np.load(f"{cache}/embeddings.npy").astype(np.float32)
        assert json.load(open(f"{cache}/embeddings_meta.json"))["ids"] == [r["id"] for r in pool], "embeddings out of date"
    pairs = {}
    for pairing in sorted({c[2] for c in cells}):
        pairs_path = f"{args.out}/pairs_{pairing}_seed{args.seed}.jsonl" if args.pool else f"{args.out}/pairs_seed{args.seed}.jsonl"
        ps = load_jsonl(pairs_path)
        if len(ps) < args.n_pairs:  # extend deterministically; same seed gives the same prefix
            ps = (make_pairs(pool, args.n_pairs, args.seed, pairing, tags, E, args.tag_conf, args.band, args.dup_cut) if args.pool
                  else sample_pairs(args.base_data, args.n_pairs, args.seed))
            with open(pairs_path, "w", encoding="utf-8") as f:
                f.writelines(json.dumps(p, ensure_ascii=False) + "\n" for p in ps)
        pairs[pairing] = ps[:args.n_pairs]
    toks = {role: AutoTokenizer.from_pretrained(m[1]) for role, m in MODELS.items()}
    path = lambda cell, stage: f"{args.out}/{cell}/{stage}.jsonl"
    keep = {f"pair_{i:05d}" for i in range(args.n_pairs)}  # downstream stages only see the first --n_pairs pairs
    load = lambda cell, stage: [r for r in load_jsonl(path(cell, stage)) if r["pair_id"] in keep]
    for _, cell, _ in cells:
        os.makedirs(f"{args.out}/{cell}", exist_ok=True)

    def todo(cell, stage, items, key=lambda x: x["pair_id"]):
        done = {key(r) for r in load(cell, stage)}
        return [x for x in items if key(x) not in done]

    print("== fuse")
    for arm, cell, pairing in cells:
        for p in todo(cell, "fuse", pairs[pairing]):
            lines = topic_lines(arm, p, tags, slots, args.tag_conf, args.seed)
            prompt = fuse_prompt(arm, p["sources"], toks["fuser"], hint, lines)
            g = generate("fuser", prompt, req_seed(args.seed, "fuse", cell, p["pair_id"]))
            append_jsonl(path(cell, "fuse"), {"pair_id": p["pair_id"], "arm": arm, "pairing": pairing,
                                              "topic_lines": lines, "prompt": prompt, **g,
                                              "problem": parse_fused(g["text"])})
            print(f"[fuse {cell}] {p['pair_id']} {g['gen_tokens']} tok {g['seconds']}s")

    print("== formalize")
    for arm, cell, pairing in cells:
        fused = [r for r in load(cell, "fuse") if r["problem"] is not None]
        for r in todo(cell, "formalize", fused):
            prompt = PromptWriter.formulate_prompt_autoformalization_Goedel("thm_GAR", r["problem"], toks["formalizer"])
            g = generate("formalizer", prompt, req_seed(args.seed, "formalize", cell, r["pair_id"]))
            append_jsonl(path(cell, "formalize"), {"pair_id": r["pair_id"], "problem": r["problem"], "prompt": prompt, **g,
                                                  "statement": parse_formalized(g["text"])})
            print(f"[formalize {cell}] {r['pair_id']} {g['gen_tokens']} tok {g['seconds']}s")

    print("== check")
    lc = LeanChecker(args.lean_url)
    for arm, cell, pairing in cells:
        recs = [r for r in todo(cell, "check", load(cell, "formalize")) if r["statement"] is not None]
        if recs:
            for r, res in zip(recs, lc.verify_statement([r["statement"] for r in recs])):
                blocks = re.findall(r'(theorem.*?:= by)', r["statement"], re.DOTALL)
                append_jsonl(path(cell, "check"), {"pair_id": r["pair_id"], "NL": r["problem"], "valid": res["pass"],
                                                  "Statement": blocks[-1] if blocks else None, "messages": res["verify_result"]})

    print("== filter")
    for arm, cell, pairing in cells:
        for r in todo(cell, "filter", [r for r in load(cell, "check") if r["valid"]]):
            g, parts, groups = glued(r["Statement"])
            append_jsonl(path(cell, "filter"), {"pair_id": r["pair_id"], "glued": g, "analysis_leak": analysis_leak(r["NL"]),
                                               "parts": parts, "groups": groups})

    print("== prove")
    for arm, cell, pairing in cells:
        bad = {f["pair_id"] for f in load(cell, "filter") if f["glued"] or f["analysis_leak"]} if args.prefilter else set()
        valid = [r for r in load(cell, "check") if r["valid"] and r["pair_id"] not in bad]
        items = [dict(r, attempt=k) for r in valid for k in range(args.proofs)]
        for it in todo(cell, "prove", items, key=lambda x: (x["pair_id"], x["attempt"])):
            prompt = PromptWriter.formulate_prompt_proof_writing_Goedel_V2(it["NL"], it["Statement"], toks["prover"])
            # The GGUF has add_bos_token=true, so the server prepends BOS itself; drop the template's BOS to avoid a double BOS
            prompt = prompt.removeprefix(toks["prover"].bos_token or "")
            g = generate("prover", prompt, req_seed(args.seed, "prove", cell, it["pair_id"], it["attempt"]))
            blocks = utils.extract_code_blocks_as_list(g["text"], code_type="lean4")
            append_jsonl(path(cell, "prove"), {"pair_id": it["pair_id"], "attempt": it["attempt"], **g,
                                              "proof": blocks[-1] if blocks else None})
            print(f"[prove {cell}] {it['pair_id']}#{it['attempt']} {g['gen_tokens']} tok {g['seconds']}s trunc={g['truncated']}")

    print("== verify")
    # v2 of the nocomment scorer stops counting incomplete blocks as modifications; v1 files are kept as they were
    verify_stage = "verify" if args.mod_check == "gar" else f"verify_{args.mod_check}_v2"
    for arm, cell, pairing in cells:
        statements = {r["pair_id"]: r["Statement"] for r in load(cell, "check") if r["valid"]}
        attempts = load(cell, "prove")
        for pid in sorted({a["pair_id"] for a in attempts} - {r["pair_id"] for r in load(cell, verify_stage)}):
            mine = [a for a in attempts if a["pair_id"] == pid]
            if len(mine) < args.proofs:
                continue  # proving for this statement is not finished yet
            proofs = [a["proof"] for a in mine if a["proof"] is not None]  # GAR drops attempts with no lean4 block
            flags, modified, incomplete, brute = (score_statement(lc, pid, statements[pid], proofs, args.mod_check,
                                                                  FORBID_BRUTE_FORCE[args.domain])
                                                  if proofs else ([], 0, 0, 0))
            passes = sum(flags)
            append_jsonl(path(cell, verify_stage), {
                "pair_id": pid, "n_attempts": len(mine), "n_extracted": len(proofs), "passes": passes,
                "p_gar": passes / len(proofs) if proofs else 0.0,  # GAR's denominator: extracted proofs only
                "p_all": passes / len(mine), "m": modified / len(proofs) if proofs else 0.0,
                "incomplete": incomplete / len(proofs) if proofs else 0.0,
                "brute_force": brute / len(proofs) if proofs else 0.0,
                "pass_flags": flags})

    print("== summary")
    summary = {cell: summarize(load(cell, "fuse"), load(cell, "formalize"), load(cell, "check"),
                               load(cell, "prove"), load(cell, verify_stage), args.n_pairs, load(cell, "filter"))
               for arm, cell, pairing in cells}
    summary_name = "summary.json" if args.mod_check == "gar" else f"summary_{args.mod_check}_v2.json"
    with open(f"{args.out}/{summary_name}", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
