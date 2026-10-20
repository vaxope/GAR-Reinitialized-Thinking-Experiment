"""
Formalization faithfulness for the puzzle domain. The puzzle generator's reference answer is used only here, to check
whether the formalizer's Lean statement says the right thing; solvers are always verified by Lean.

For each source puzzle with a reference answer, the formalizer gets the puzzle plus "The answer is: <answer>" and must
state it as a theorem. Then:
  compiles   the statement elaborates (`sorry` proof)
  provable   automation (decide / native_decide / norm_num / simp / omega) proves it   -> consistent with the answer
  refuted    the same automation proves its negation                                  -> unfaithful
  unknown    neither (too large for brute force, or not decidable as written)
Brute force is allowed here because this is our check, not the solver.
"""
import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter, defaultdict

from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import PromptWriter  # noqa: E402
from LeanChecker import LeanChecker, Lean4_headers  # noqa: E402
from exp001_local import MODELS, generate, load_jsonl, append_jsonl, parse_formalized, req_seed  # noqa: E402
from filters import split_statement  # noqa: E402

AUTO = "by first | decide | native_decide | norm_num | simp | omega"


def lean_ok(lc, code, timeout=120):
    cid = hashlib.sha256(code.encode()).hexdigest()
    r = lc._submit_job([{"proof": Lean4_headers + code, "custom_id": cid}], timeout=timeout)["results"][0]
    if r["error"] is not None or r["response"] is None:
        return False
    return not any(m["severity"] == "error" for m in r["response"].get("messages", []))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="tagging/puzzles/pool.jsonl")
    ap.add_argument("--per_family", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=30, help="seconds per Lean check; brute force on big puzzles times out")
    ap.add_argument("--out", default="exp_results/puzzle_faithfulness/records.jsonl")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    by = defaultdict(list)
    for r in load_jsonl(a.pool):
        if r["meta"].get("answer") is not None:
            by[r["meta"]["family"]].append(r)
    rng = random.Random(a.seed)
    items = []
    for fam in sorted(by):  # shuffled prefix, so a larger --per_family extends a smaller one
        xs = list(by[fam])
        rng.shuffle(xs)
        items += xs[:a.per_family]
    tok = AutoTokenizer.from_pretrained(MODELS["formalizer"][1])
    lc = LeanChecker()
    done = {r["id"] for r in load_jsonl(a.out)}
    for r in items:
        if r["id"] in done:
            continue
        nl = f"{r['text']}\n\nThe answer is: {r['meta']['answer']}"
        prompt = PromptWriter.formulate_prompt_autoformalization_Goedel("thm_GAR", nl, tok)
        g = generate("formalizer", prompt, req_seed(a.seed, "faith", r["id"]))
        st = parse_formalized(g["text"])
        rec = {"id": r["id"], "family": r["meta"]["family"], "answer": r["meta"]["answer"], "gen_tokens": g["gen_tokens"],
               "statement": st, "compiles": False, "provable": False, "refuted": False}
        sp = split_statement(st) if st else None
        if st and sp:
            binders, concl = sp
            rec["compiles"] = lean_ok(lc, f"theorem chk {binders} : {concl} := by sorry", a.timeout)
            if rec["compiles"]:
                rec["provable"] = lean_ok(lc, f"theorem chk {binders} : {concl} := {AUTO}", a.timeout)
                if not rec["provable"]:
                    rec["refuted"] = lean_ok(lc, f"theorem chk : ¬ (∀ {binders}, {concl}) := {AUTO}" if binders
                                             else f"theorem chk : ¬ ({concl}) := {AUTO}", a.timeout)
        append_jsonl(a.out, rec)
        print({k: v for k, v in rec.items() if k not in ("statement", "answer")}, flush=True)
    wanted = {r["id"] for r in items}
    recs = [x for x in load_jsonl(a.out) if x["id"] in wanted]
    summ = {}
    for fam in sorted({x["family"] for x in recs}) + ["ALL"]:
        xs = [x for x in recs if fam in ("ALL", x["family"])]
        c = Counter("provable" if x["provable"] else "refuted" if x["refuted"] else "compiles_unknown" if x["compiles"]
                    else "no_compile" for x in xs)
        summ[fam] = {"n": len(xs), **c}
    with open(os.path.join(os.path.dirname(a.out), "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summ, f, indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
