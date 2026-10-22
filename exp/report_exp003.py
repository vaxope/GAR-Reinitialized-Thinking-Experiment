"""
EXP-003 option G report: GAR's maths-specific reinitialized-thinking guidance (C) vs the domain-agnostic skeleton (D1) vs
skeleton + topic checks (D2), in maths, puzzles and STEM. CPU only; reads the run files; writes a markdown report.

Per domain x arm: fused pairs, compile rate, clean statements, statements proved (4 attempts, 16k), useful (0 < p <= 1/2),
too easy, too hard, useful per fused pair, total tokens per useful problem. Fisher exact tests vs C on compile rate and on
useful-per-statement; pooled across domains as well (stratification ignored, so the pooled test is only indicative).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_arms import fisher_two_sided, wilson  # noqa: E402

RUNS = {"math": 28, "stem": 40, "puzzles": 50}  # pairs per arm whose statements go to the prover in pass 2
FUSED = {"math": 100, "stem": 40, "puzzles": 50}
ARMS = [("C", "C_random"), ("D1", "D1_random"), ("D2", "D2_random")]
VERIFY = "verify"  # or verify_nocomment_v2 (comment-insensitive scorer)


def L(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")] if os.path.exists(path) else []


def cell(D, c):
    base = f"exp_results/EXP-003-{D}/{c}"
    keep = lambda r, n: int(r["pair_id"][-5:]) < n
    fuse = [r for r in L(f"{base}/fuse.jsonl") if keep(r, FUSED[D])]
    form = [r for r in L(f"{base}/formalize.jsonl") if keep(r, FUSED[D])]
    check = [r for r in L(f"{base}/check.jsonl") if keep(r, FUSED[D])]
    filt = {r["pair_id"]: r for r in L(f"{base}/filter.jsonl")}
    ver = L(f"{base}/{VERIFY}.jsonl")
    att = L(f"{base}/prove.jsonl")
    proved_pairs = {v["pair_id"] for v in ver}
    pairs_covered = max([int(p[-5:]) + 1 for p in proved_pairs] + [0])  # prefix of pairs whose statements were proved
    n_pairs_proved = min(RUNS[D], max(pairs_covered, 1)) if ver else 0
    useful = [v for v in ver if 0 < v["p_gar"] <= 0.5]
    tok_fuser = sum(r["gen_tokens"] for r in fuse if keep(r, n_pairs_proved))
    tok_form = sum(r["gen_tokens"] for r in form if keep(r, n_pairs_proved))
    tok_prover = sum(a["gen_tokens"] for a in att if a["pair_id"] in proved_pairs)
    valid = [r for r in check if r["valid"]]
    clean = [r for r in valid if not (filt.get(r["pair_id"], {}).get("glued") or filt.get(r["pair_id"], {}).get("analysis_leak"))]
    return {"fused": len(fuse), "compiles": len(valid), "clean": len(clean), "proved": len(ver), "useful": len(useful),
            "easy": sum(v["p_gar"] > 0.5 for v in ver), "hard": sum(v["p_gar"] == 0 for v in ver),
            "pairs_proved": n_pairs_proved, "tokens": tok_fuser + tok_form + tok_prover,
            "brute": sum(v.get("brute_force", 0) for v in ver) / max(len(ver), 1)}


def main():
    global VERIFY
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", default="verify", choices=["verify", "verify_nocomment_v2"])
    VERIFY = ap.parse_args().verify
    out = ["| Domain | Arm | Compiles (of fused) | Clean | Proved | Useful | Too easy | Too hard | Useful / proved [95% CI] "
           "| p vs C (useful) | Tokens per useful |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    pooled = {a: [0, 0] for a, _ in ARMS}
    data = {}
    for D in RUNS:
        cs = {a: cell(D, c) for a, c in ARMS}
        data[D] = cs
        b = cs["C"]
        for a, s in cs.items():
            lo, hi = wilson(s["useful"], s["proved"]) if s["proved"] else (0, 0)
            pc = fisher_two_sided(s["compiles"], s["fused"] - s["compiles"], b["compiles"], b["fused"] - b["compiles"])
            pu = fisher_two_sided(s["useful"], s["proved"] - s["useful"], b["useful"], b["proved"] - b["useful"])
            tpu = f"{s['tokens'] / s['useful'] / 1000:.0f}k" if s["useful"] else "–"
            out.append(f"| {D} | {a} | {s['compiles']}/{s['fused']} ({s['compiles'] / max(s['fused'], 1):.2f}"
                       f"{'' if a == 'C' else f', p={pc:.3f}'}) | {s['clean']} | {s['proved']} | {s['useful']} | {s['easy']} "
                       f"| {s['hard']} | {s['useful'] / max(s['proved'], 1):.2f} [{lo:.2f}, {hi:.2f}] "
                       f"| {'–' if a == 'C' else f'{pu:.2f}'} | {tpu} |")
            pooled[a][0] += s["useful"]
            pooled[a][1] += s["proved"]
    out += ["", "| Pooled (all domains) | Useful / proved | Rate [95% CI] | p vs C |", "|---|---|---|---|"]
    for a, (u, n) in pooled.items():
        lo, hi = wilson(u, n) if n else (0, 0)
        p = fisher_two_sided(u, n - u, pooled["C"][0], pooled["C"][1] - pooled["C"][0])
        out.append(f"| {a} | {u}/{n} | {u / max(n, 1):.2f} [{lo:.2f}, {hi:.2f}] | {'–' if a == 'C' else f'{p:.2f}'} |")
    text = "\n".join(out)
    tag = "" if VERIFY == "verify" else "-nocomment"
    with open(f"exp_results/EXP-003-report{tag}.md", "w", encoding="utf-8") as f:
        f.write(text + "\n")
    with open(f"exp_results/EXP-003-report{tag}.json", "w", encoding="utf-8") as f:
        json.dump({"cells": data, "pooled": pooled}, f, indent=1)
    sys.stdout.reconfigure(encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
