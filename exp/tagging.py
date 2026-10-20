"""
Domain-agnostic topic tagging (CLAUDE.md §2). The same stages run on any pool of {"id", "text", "meta"} rows:

  embed      qwen3-embedding (Ollama) -> L2-normalised float16 matrix, cached outside the repo
  cluster    k-means on the embeddings
  name       an LLM names each cluster from its most central problems, then merges clusters into <= --max_topics topics
  label      the LLM labels --n_label problems by single-forward-pass logit scoring over topic letters (top-2 + confidence)
  propagate  embedding kNN spreads the LLM labels to the whole pool; a held-out split measures kNN vs LLM agreement
  validate   200-row CSV for hand-checking, plus a cross-tab against any label already in meta (a sanity check only)
  slots      the LLM drafts 3 candidate <analysis> slots per topic: pitfalls and checks, never "use method X"

Only --desc (one line describing the pool) and --meta_key are domain-specific.
Every stage caches its output and is skipped if the output exists (pass --force to recompute).
"""
import argparse
import csv
import json
import math
import os
import random
import re
import sys
import time

import numpy as np
import requests

OLLAMA = "http://127.0.0.1:11434"
EMBED_MODEL = "qwen3-embedding:0.6b"
LLM_MODEL = "qwen3:8b"
EMBED_INSTRUCT = "Instruct: Identify the topic of this problem\nQuery: "
LETTERS = "ABCDEFGHIJKLMNOPQRST"  # Ollama returns at most 20 top logprobs, so at most 20 topics


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)


def chat(user, num_predict=400, temperature=0.0, seed=0, logprobs=0):
    """Qwen3-8B in non-thinking mode (empty think block), raw prompt so nothing is added by Ollama's template."""
    prompt = f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    return _gen(prompt, num_predict, temperature, seed, logprobs)


def _gen(prompt, num_predict, temperature, seed, logprobs=0):
    body = {"model": LLM_MODEL, "prompt": prompt, "raw": True, "stream": False, "keep_alive": "30m",
            "options": {"num_predict": num_predict, "temperature": temperature, "seed": seed, "top_k": 0, "top_p": 1.0,
                        "min_p": 0.0, "repeat_penalty": 1.0, "num_ctx": 8192}}
    if logprobs:
        body.update(logprobs=True, top_logprobs=logprobs)
    r = requests.post(f"{OLLAMA}/api/generate", json=body, timeout=600)
    r.raise_for_status()
    return r.json()


def letter_logprobs(top, letters):
    """Sum the probability of every top token that spells a letter ('D', ' D', 'D)'); variants must add, not overwrite."""
    p = {}
    for t in top:
        L = t["token"].strip().rstrip(")").strip()
        if L in letters:
            p[L] = p.get(L, 0.0) + math.exp(t["logprob"])
    return p


def parse_json(text):
    m = re.search(r"(\{.*\}|\[.*\])", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


class Tagger:
    def __init__(self, a):
        self.a = a
        self.rows = load_jsonl(a.pool)
        self.ids = [r["id"] for r in self.rows]
        os.makedirs(a.out, exist_ok=True)
        os.makedirs(a.cache, exist_ok=True)
        self.p = lambda name: os.path.join(a.out, name)

    def done(self, path):
        return os.path.exists(path) and not self.a.force

    # ---- embed ----
    def embed(self):
        path = os.path.join(self.a.cache, "embeddings.npy")
        if self.done(path):
            return
        vecs = []
        t0 = time.time()
        for i in range(0, len(self.rows), 64):
            batch = [EMBED_INSTRUCT + r["text"][:2000] for r in self.rows[i:i + 64]]
            r = requests.post(f"{OLLAMA}/api/embed", json={"model": EMBED_MODEL, "input": batch, "keep_alive": "30m"},
                              timeout=600)
            r.raise_for_status()
            vecs += r.json()["embeddings"]
            if i % 3200 == 0:
                print(f"[embed] {i}/{len(self.rows)} {time.time() - t0:.0f}s", flush=True)
        E = np.asarray(vecs, dtype=np.float32)
        E /= np.linalg.norm(E, axis=1, keepdims=True)
        np.save(path, E.astype(np.float16))
        save_json({"ids": self.ids, "model": EMBED_MODEL, "instruct": EMBED_INSTRUCT, "seconds": round(time.time() - t0)},
                  os.path.join(self.a.cache, "embeddings_meta.json"))

    def E(self):
        return np.load(os.path.join(self.a.cache, "embeddings.npy")).astype(np.float32)

    # ---- cluster ----
    def cluster(self):
        if self.done(self.p("clusters.json")):
            return
        from sklearn.cluster import KMeans
        E = self.E()
        km = KMeans(n_clusters=self.a.k, n_init=4, random_state=self.a.seed).fit(E)
        C = km.cluster_centers_ / np.linalg.norm(km.cluster_centers_, axis=1, keepdims=True)
        examples = {}
        for c in range(self.a.k):
            idx = np.where(km.labels_ == c)[0]
            order = idx[np.argsort(-(E[idx] @ C[c]))]
            examples[str(c)] = [self.ids[i] for i in order[:12]]
        save_json({"k": self.a.k, "labels": km.labels_.tolist(), "sizes": np.bincount(km.labels_).tolist(),
                   "examples": examples}, self.p("clusters.json"))

    # ---- name ----
    def name(self):
        if self.done(self.p("topics.json")):
            return
        cl = json.load(open(self.p("clusters.json"), encoding="utf-8"))
        text = {r["id"]: r["text"] for r in self.rows}
        names = {}
        if os.path.exists(self.p("cluster_names.json")) and not self.a.force:
            names = json.load(open(self.p("cluster_names.json"), encoding="utf-8"))
        for c, ex in cl["examples"].items():
            if c in names:
                continue
            sample = "\n\n".join(f"- {text[i][:400]}" for i in ex)
            for attempt in range(3):
                out = chat(f"These are {self.a.desc}. They were grouped together because they are similar.\n\n{sample}\n\n"
                           "Give a short topic name (1-4 words) for WHAT these problems are about (their subject matter), "
                           "not how to solve them, and a one-sentence description.\n"
                           'Reply only with JSON: {"name": "...", "description": "..."}', seed=attempt)
                j = parse_json(out["response"])
                if isinstance(j, dict) and j.get("name"):
                    names[c] = {"name": j["name"].strip(), "description": str(j.get("description", "")).strip(),
                                "size": cl["sizes"][int(c)]}
                    break
            print(f"[name] cluster {c}: {names.get(c)}", flush=True)
        save_json(names, self.p("cluster_names.json"))
        # Merge in two easy steps instead of one hard JSON task: propose <= max_topics topics, then assign each cluster
        # to one of them by logit scoring (the same single-forward-pass mechanism as the label stage)
        listing = "\n".join(f"- {v['name']}: {v['description']} ({v['size']} problems)" for v in names.values())
        limit = min(self.a.max_topics, len(LETTERS))
        proposed = None
        for attempt in range(5):
            out = chat(f"These are topic clusters of {self.a.desc}:\n{listing}\n\n"
                       f"Group them into at most {limit} broader topics so that clusters about the same subject share a "
                       "topic. Topic names must be short (1-4 words) and describe subject matter, not solution methods.\n"
                       'Reply only with a JSON list: [{"name": "...", "description": "..."}, ...]',
                       num_predict=1500, seed=attempt)
            j = parse_json(out["response"])
            if isinstance(j, list) and 2 <= len(j) <= limit and all(isinstance(t, dict) and t.get("name") for t in j):
                proposed = [{"name": t["name"].strip(), "description": str(t.get("description", "")).strip()} for t in j]
                break
            print(f"[name] proposal attempt {attempt} invalid, retrying", flush=True)
        assert proposed, "could not get a valid topic proposal"
        letters = LETTERS[:len(proposed)]
        menu = "\n".join(f"({L}) {t['name']}: {t['description']}" for L, t in zip(letters, proposed))
        for t in proposed:
            t["clusters"] = []
        for c, v in names.items():
            ex = "\n".join(f"- {text[i][:200]}" for i in cl["examples"][c][:3])
            prompt = (f"<|im_start|>user\nWhich topic best describes this cluster of {self.a.desc}?\n\nTopics:\n{menu}\n\n"
                      f"Cluster: {v['name']} - {v['description']}\nExamples:\n{ex}\n\nReply with the letter of the best topic."
                      f"<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nAnswer: (")
            lp = letter_logprobs(_gen(prompt, 1, 0.0, 0, logprobs=20)["logprobs"][0]["top_logprobs"], letters)
            best = max(lp, key=lp.get)
            proposed[letters.index(best)]["clusters"].append(int(c))
        topics = [t for t in proposed if t["clusters"]]
        for t in topics:
            print(f"[name] topic {t['name']}: clusters {t['clusters']}", flush=True)
        save_json({"topics": topics}, self.p("topics.json"))

    def topics(self):
        return json.load(open(self.p("topics.json"), encoding="utf-8"))["topics"]

    # ---- label (single forward pass, logit scoring) ----
    def label(self):
        path = self.p("llm_labels.jsonl")
        topics = self.topics()
        letters = LETTERS[:len(topics)]
        menu = "\n".join(f"({L}) {t['name']}: {t['description']}" for L, t in zip(letters, topics))
        head = (f"Classify the main topic of the problem below: what it is about, not how to solve it. "
                f"The problems are {self.a.desc}.\n\nTopics:\n{menu}\n\nProblem:\n")
        rng = random.Random(self.a.seed)
        sample = rng.sample(range(len(self.rows)), min(self.a.n_label, len(self.rows)))
        done = {r["id"] for r in load_jsonl(path)} if os.path.exists(path) and not self.a.force else set()
        if self.a.force and os.path.exists(path):
            os.remove(path)
        t0 = time.time()
        with open(path, "a", encoding="utf-8") as f:
            for n, i in enumerate(sample):
                r = self.rows[i]
                if r["id"] in done:
                    continue
                prompt = (f"<|im_start|>user\n{head}{r['text'][:2500]}\n\nReply with the letter of the best topic."
                          f"<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nAnswer: (")
                out = _gen(prompt, 1, 0.0, 0, logprobs=20)
                lp = letter_logprobs(out["logprobs"][0]["top_logprobs"], letters)
                z = sum(lp.values()) or 1.0
                probs = {topics[letters.index(L)]["name"]: v / z for L, v in lp.items()}
                f.write(json.dumps({"id": r["id"], "probs": probs, "mass": round(z, 4)}, ensure_ascii=False) + "\n")
                f.flush()  # keep the file resumable if the run is interrupted
                if n % 250 == 0:
                    print(f"[label] {n}/{len(sample)} {time.time() - t0:.0f}s", flush=True)

    # ---- propagate (kNN) ----
    def propagate(self):
        if self.done(self.p("tags.jsonl")):
            return
        topics = [t["name"] for t in self.topics()]
        lab = load_jsonl(self.p("llm_labels.jsonl"))
        pos = {i: n for n, i in enumerate(self.ids)}
        L_idx = np.array([pos[r["id"]] for r in lab])
        P = np.array([[r["probs"].get(t, 0.0) for t in topics] for r in lab], dtype=np.float32)
        E = self.E()

        def knn(query, ref_idx, ref_P, k=self.a.knn_k, exclude_self=False):
            out = np.zeros((len(query), len(topics)), dtype=np.float32)
            R = E[ref_idx]
            for s in range(0, len(query), 2048):
                S = E[query[s:s + 2048]] @ R.T
                if exclude_self:
                    S[S > 0.9999] = -1
                nn = np.argpartition(-S, k, axis=1)[:, :k]
                w = np.exp(np.take_along_axis(S, nn, 1) / 0.05)
                out[s:s + 2048] = np.einsum("nk,nkt->nt", w / w.sum(1, keepdims=True), ref_P[nn])
            return out

        rng = np.random.default_rng(self.a.seed)
        perm = rng.permutation(len(L_idx))
        test, train = perm[:500], perm[500:]
        Q = knn(L_idx[test], L_idx[train], P[train])
        top1 = (Q.argmax(1) == P[test].argmax(1)).mean()
        top2 = np.mean([P[test][n].argmax() in np.argsort(-Q[n])[:2] for n in range(len(test))])
        save_json({"heldout": len(test), "train": len(train), "knn_k": self.a.knn_k,
                   "top1_agreement_with_llm": round(float(top1), 4), "llm_top1_in_knn_top2": round(float(top2), 4)},
                  self.p("knn_heldout.json"))
        allP = knn(np.arange(len(self.ids)), L_idx, P)
        allP[L_idx] = P  # LLM-labelled rows keep their own scores
        src = np.array(["knn"] * len(self.ids), dtype=object)
        src[L_idx] = "llm"
        with open(self.p("tags.jsonl"), "w", encoding="utf-8") as f:
            for n, i in enumerate(self.ids):
                o = np.argsort(-allP[n])[:2]
                f.write(json.dumps({"id": i, "tags": [[topics[j], round(float(allP[n][j]), 4)] for j in o],
                                    "source": src[n]}, ensure_ascii=False) + "\n")
        print(f"[propagate] held-out kNN vs LLM: top1 {top1:.3f}, LLM top1 in kNN top2 {top2:.3f}", flush=True)

    # ---- validate ----
    def validate(self):
        tags = {r["id"]: r for r in load_jsonl(self.p("tags.jsonl"))}
        rng = random.Random(self.a.seed + 1)
        rows = rng.sample(self.rows, 200)
        with open(self.p("validation_200.csv"), "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "source", "tag1", "p1", "tag2", "p2", self.a.meta_key, "text", "tag1_correct (y/n)",
                        "tag2_correct (y/n/-)", "notes"])
            for r in rows:
                t = tags[r["id"]]
                w.writerow([r["id"], t["source"], t["tags"][0][0], t["tags"][0][1], t["tags"][1][0], t["tags"][1][1],
                            r["meta"].get(self.a.meta_key), r["text"][:600], "", "", ""])
        cross, conf = {}, []
        for r in self.rows:
            t = tags[r["id"]]["tags"][0]
            cross.setdefault(t[0], {}).setdefault(str(r["meta"].get(self.a.meta_key)), 0)
            cross[t[0]][str(r["meta"].get(self.a.meta_key))] += 1
            conf.append(t[1])
        purity = sum(max(v.values()) for v in cross.values()) / len(self.rows)
        sizes = {k: sum(v.values()) for k, v in cross.items()}
        # pairwise cosine quantiles: the similarity band for pairing is calibrated on these per domain
        E = self.E()
        sub = np.random.default_rng(self.a.seed).choice(len(E), min(2000, len(E)), replace=False)
        S = (E[sub] @ E[sub].T)[np.triu_indices(len(sub), 1)]
        cos_q = {str(q): round(float(np.quantile(S, q)), 4) for q in (0.01, 0.1, 0.5, 0.9, 0.95, 0.99)}
        save_json({"meta_key": self.a.meta_key, "purity_vs_meta": round(purity, 4), "topic_sizes": sizes,
                   "pairwise_cosine_quantiles": cos_q,
                   "confidence_quantiles": {q: round(float(np.quantile(conf, q)), 3) for q in (0.1, 0.25, 0.5, 0.75)},
                   "share_conf_below_0.5": round(float(np.mean(np.array(conf) < 0.5)), 4), "crosstab": cross},
                  self.p("validation_summary.json"))
        print(f"[validate] purity vs {self.a.meta_key}: {purity:.3f}; topic sizes {sizes}", flush=True)

    # ---- slots ----
    def slots(self):
        if self.done(self.p("slots.json")):
            return
        tags = load_jsonl(self.p("tags.jsonl"))
        text = {r["id"]: r["text"] for r in self.rows}
        out = {}
        for t in self.topics():
            ex = [r["id"] for r in tags if r["tags"][0][0] == t["name"] and r["tags"][0][1] > 0.8][:3]
            sample = "\n\n".join(f"- {text[i][:400]}" for i in ex)
            for attempt in range(3):
                o = chat(f"Topic: {t['name']} - {t['description']}\nThe problems are {self.a.desc}. Examples:\n{sample}\n\n"
                         "Someone will combine a problem from this topic with other problems into one new, harder problem. "
                         "Write 3 different short checklists (each at most 35 words) of pitfalls to avoid and checks to make "
                         "so that the new problem stays valid, well-posed and non-trivial in this topic. Phrase them as "
                         "checks (e.g. 'make sure ...', 'avoid ...'); never tell the solver which method to use.\n"
                         'Reply only with a JSON list of 3 strings.', num_predict=600, temperature=0.7, seed=attempt)
                j = parse_json(o["response"])
                if isinstance(j, list) and len(j) >= 3 and all(isinstance(s, str) for s in j[:3]):
                    out[t["name"]] = [" ".join(s.split()[:40]) for s in j[:3]]
                    break
            print(f"[slots] {t['name']}: {out.get(t['name'])}", flush=True)
        save_json(out, self.p("slots.json"))


def selftest():
    assert parse_json('x {"name": "A", "description": "b"} y') == {"name": "A", "description": "b"}
    assert parse_json("[1, 2]") == [1, 2] and parse_json("no json") is None
    # a lower-probability ' D' variant must not overwrite 'D' (the bug that hit 20% of the first maths labels)
    lp = letter_logprobs([{"token": "D", "logprob": 0.0}, {"token": " D", "logprob": -25.0}, {"token": "x", "logprob": -1}],
                         "ABCD")
    assert abs(lp["D"] - 1.0) < 1e-6 and set(lp) == {"D"}
    print("tagging selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stages", nargs="+", choices=["embed", "cluster", "name", "label", "propagate", "validate", "slots",
                                                  "all", "selftest"])
    ap.add_argument("--domain", required=False, default="math")
    ap.add_argument("--pool", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--cache", default=None, help="embedding cache (outside the repo; ~2 KB per problem)")
    ap.add_argument("--desc", default="competition mathematics problems")
    ap.add_argument("--meta_key", default="Problem_type")
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--max_topics", type=int, default=15)
    ap.add_argument("--n_label", type=int, default=5000)
    ap.add_argument("--knn_k", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if "selftest" in a.stages:
        selftest()
        sys.exit()
    a.pool = a.pool or f"tagging/{a.domain}/pool.jsonl"
    a.out = a.out or f"tagging/{a.domain}"
    a.cache = a.cache or os.path.join(os.path.expanduser("~"), "GAR-cache", a.domain)
    sys.stdout.reconfigure(encoding="utf-8")
    t = Tagger(a)
    stages = ["embed", "cluster", "name", "label", "propagate", "validate", "slots"] if "all" in a.stages else a.stages
    save_json({**vars(a), "embed_model": EMBED_MODEL, "llm": LLM_MODEL, "time": int(time.time())},
              os.path.join(a.out, f"config_{'_'.join(stages)}.json"))
    for s in stages:
        print(f"== {s}", flush=True)
        getattr(t, s)()
