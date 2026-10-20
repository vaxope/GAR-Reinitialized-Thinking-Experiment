"""
Compare fuser arms of one EXP-001 run with uncertainty. CPU only; reads the run's JSONL files.

For each scoring (GAR's check and the comment-insensitive v2 check) and each arm: valid statements, useful problems
(0 < p <= 0.5), clean useful problems (also not glued / no <analysis> leak), yield with a Wilson 95% interval, and a
two-sided Fisher exact test of each arm's clean yield against arm C.
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp001_local import load_jsonl  # noqa: E402


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def fisher_two_sided(a, b, c, d):
    """2x2 table [[a, b], [c, d]]; sums hypergeometric probabilities <= the observed one."""
    n1, n2, k = a + b, c + d, a + c
    pmf = lambda x: math.comb(n1, x) * math.comb(n2, k - x) / math.comb(n1 + n2, k)
    obs = pmf(a)
    return min(1.0, sum(pmf(x) for x in range(max(0, k - n2), min(k, n1) + 1) if pmf(x) <= obs * (1 + 1e-9)))


def arm_stats(run, arm, n_pairs, verify_file):
    keep = {f"pair_{i:05d}" for i in range(n_pairs)}
    L = lambda stage: [r for r in load_jsonl(f"{run}/{arm}/{stage}.jsonl") if r["pair_id"] in keep]
    bad = {f["pair_id"] for f in L("filter") if f["glued"] or f["analysis_leak"]}
    ver = L(verify_file)
    useful = [v["pair_id"] for v in ver if 0 < v["p_gar"] <= 0.5]
    clean = [p for p in useful if p not in bad]
    return {"valid": sum(r["valid"] for r in L("check")), "verified": len(ver), "solved_any": sum(v["passes"] > 0 for v in ver),
            "useful": len(useful), "useful_clean": len(clean), "useful_clean_ids": clean}


def selftest():
    assert abs(wilson(0, 15)[1] - 0.2039) < 1e-3 and wilson(15, 15)[1] == 1.0
    assert abs(fisher_two_sided(1, 14, 1, 14) - 1.0) < 1e-9
    assert abs(fisher_two_sided(5, 0, 0, 5) - 2 / 252) < 1e-9
    print("compare selftest ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="exp_results/EXP-001-local-v4-goedel16k")
    ap.add_argument("--arms", nargs="+", default=["C", "D1", "D4"])
    ap.add_argument("--n_pairs", type=int, default=15)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    out = {}
    lines = []
    for label, vf in [("GAR check", "verify"), ("comment-insensitive v2", "verify_nocomment_v2")]:
        stats = {a: arm_stats(args.run, a, args.n_pairs, vf) for a in args.arms}
        lines += [f"\n### {label} (n = {args.n_pairs} pairs per arm)",
                  "| Arm | Valid | Solved ≥1 | Useful | Useful, clean | Clean yield [95% CI] | Fisher p vs C |",
                  "|---|---|---|---|---|---|---|"]
        base = stats[args.arms[0]]
        for a, s in stats.items():
            lo, hi = wilson(s["useful_clean"], args.n_pairs)
            p = fisher_two_sided(s["useful_clean"], args.n_pairs - s["useful_clean"],
                                 base["useful_clean"], args.n_pairs - base["useful_clean"])
            s.update(yield_clean=s["useful_clean"] / args.n_pairs, ci95=[lo, hi], fisher_p_vs_first=p)
            lines.append(f"| {a} | {s['valid']} | {s['solved_any']} | {s['useful']} | {s['useful_clean']} | "
                         f"{s['yield_clean']:.2f} [{lo:.2f}, {hi:.2f}] | {'–' if a == args.arms[0] else f'{p:.2f}'} |")
        out[label] = stats
    with open(f"{args.run}/compare_arms.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
