"""
Domain drift of fused problems: does a fused puzzle (or STEM question) stay in its domain, or turn into a maths problem?
Each fused problem is embedded like the pools (same model and instruction); its 15 nearest neighbours in the union of
all domain pools vote on its domain. Reports, per cell, the share of fused problems whose majority vote is another domain.
"""
import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tagging import EMBED_INSTRUCT, EMBED_MODEL, OLLAMA  # noqa: E402


def embed(texts):
    out = []
    for i in range(0, len(texts), 64):
        r = requests.post(f"{OLLAMA}/api/embed", json={"model": EMBED_MODEL, "keep_alive": "10m",
                                                     "input": [EMBED_INSTRUCT + t[:2000] for t in texts[i:i + 64]]},
                          timeout=600)
        r.raise_for_status()
        out += r.json()["embeddings"]
    E = np.asarray(out, dtype=np.float32)
    return E / np.linalg.norm(E, axis=1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", nargs="+", default=["math", "puzzles"])
    ap.add_argument("--run", required=True, help="e.g. exp_results/EXP-003-puzzles")
    ap.add_argument("--domain", required=True, help="the run's own domain")
    ap.add_argument("--cells", nargs="+", default=["C_random", "D1_random", "D2_random"])
    ap.add_argument("--k", type=int, default=15)
    a = ap.parse_args()
    refs, labels = [], []
    for d in a.domains:
        E = np.load(os.path.join(os.path.expanduser("~"), "GAR-cache", d, "embeddings.npy")).astype(np.float32)
        refs.append(E)
        labels += [d] * len(E)
    R, labels = np.concatenate(refs), np.array(labels)
    out = {}
    for c in a.cells:
        fused = [json.loads(l) for l in open(f"{a.run}/{c}/fuse.jsonl", encoding="utf-8")]
        fused = [f for f in fused if f["problem"]]
        Q = embed([f["problem"] for f in fused])
        votes = []
        for s in range(0, len(Q), 256):
            S = Q[s:s + 256] @ R.T
            nn = np.argpartition(-S, a.k, axis=1)[:, :a.k]
            votes += [Counter(labels[row]).most_common(1)[0][0] for row in nn]
        cnt = Counter(votes)
        out[c] = {"n": len(votes), "votes": dict(cnt), "drift_rate": round(1 - cnt.get(a.domain, 0) / len(votes), 4),
                  "per_pair": {f["pair_id"]: v for f, v in zip(fused, votes)}}
        print(c, {k: v for k, v in out[c].items() if k != "per_pair"}, flush=True)
    with open(f"{a.run}/domain_drift.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
