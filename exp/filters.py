"""
Pre-prover filters for fused statements (Lean domain).

- analysis_leak: the fuser's <analysis> text was extracted as the problem (the model wrapped its analysis in a ```md block).
- glued: structural glued-problem check (CLAUDE.md §3). The Lean conclusion is split at top-level ↔ / ∨ / ∧ (after peeling
  binders, ∀, let and → premises). Variables that appear together in a binder type, premise or let are linked. The statement
  is glued if the parts fall into two or more groups that share no linked variable.
  shortcut: text-level parser, so it can be fooled by unusual syntax and misses semantic glue through shared constants
  (e.g. `let m := gcd 18 24 ... ∧ m = 6`); upgrade to a Lean-elaborated check via the Kimina server if it misfires.
"""
import re

from utils import utils

OPEN, CLOSE = "([{⟨", ")]}⟩"
QUANT = ("∀", "∃", "fun ", "λ", "let ")
IDENT = re.compile(r"[A-Za-z_Ͱ-Ͽ][\w₀-ₜ'!?]*")


def analysis_leak(problem):
    return problem is not None and ("<analysis>" in problem or "</analysis>" in problem)


def _split_top(s, sep):
    """Split s at depth-0 occurrences of sep, stopping at a depth-0 quantifier (its body extends to the end)."""
    parts, depth, start, i = [], 0, 0, 0
    while i < len(s):
        c = s[i]
        if c in OPEN:
            depth += 1
        elif c in CLOSE:
            depth -= 1
        elif depth == 0 and any(s.startswith(q, i) for q in QUANT) and (i == 0 or not (s[i - 1].isalnum() or s[i - 1] == "_")):
            break
        elif depth == 0 and s.startswith(sep, i):
            parts.append(s[start:i])
            start = i + len(sep)
            i = start
            continue
        i += 1
    parts.append(s[start:])
    return [p.strip() for p in parts]


def _strip_parens(s):
    s = s.strip()
    while s and s[0] in OPEN and s[-1] == CLOSE[OPEN.index(s[0])]:
        depth = 0
        for i, c in enumerate(s):
            depth += c in OPEN
            depth -= c in CLOSE
            if depth == 0 and i < len(s) - 1:
                return s  # the first bracket closes early, e.g. "(a) ∧ (b)"
        s = s[1:-1].strip()
    return s


def _binder_groups(s):
    """Contents of the depth-1 bracket groups in s, e.g. '(a b : ℝ) (h : 0 < a)' -> ['a b : ℝ', 'h : 0 < a']."""
    groups, depth, start = [], 0, None
    for i, c in enumerate(s):
        if c in OPEN:
            depth += 1
            if depth == 1:
                start = i + 1
        elif c in CLOSE:
            depth -= 1
            if depth == 0 and start is not None:
                groups.append(s[start:i])
    return groups


class _Vars:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def declare(self, names):
        for n in names:
            self.parent.setdefault(n, n)

    def mentioned(self, text):
        return {t for t in IDENT.findall(text) if t in self.parent}

    def link(self, names):
        roots = [self.find(n) for n in names]
        for r in roots[1:]:
            self.parent[r] = roots[0]


def _add_binder(V, group):
    if ":" in group and not group.lstrip().startswith(":"):
        names, typ = group.split(":", 1)
        names = [n for n in names.split() if IDENT.fullmatch(n)]
    else:
        names, typ = [], group  # instance binder like [Fintype α]
    V.declare(names)
    typ_vars = sorted(V.mentioned(typ))
    for n in names:  # `(x y : ℝ)` declares independent variables; only the type links a name to others
        V.link([n] + typ_vars)
    if not names and typ_vars:
        V.link(typ_vars)


def _parts(E, V):
    """Peel ∀ / let / premises off E and return the top-level parts of its conclusion."""
    E = _strip_parens(E)
    if E.startswith("∀"):
        body_at = None
        depth = 0
        for i, c in enumerate(E):
            depth += c in OPEN
            depth -= c in CLOSE
            if depth == 0 and c == ",":
                body_at = i
                break
        if body_at is None:
            return [E]
        binders = E[1:body_at].strip()
        if binders and binders[0] in OPEN:
            for g in _binder_groups(binders):
                _add_binder(V, g)
        else:
            _add_binder(V, binders if ":" in binders else binders + " : _")
        return _parts(E[body_at + 1:], V)
    if E.startswith("let "):
        m = re.match(r"let\s+([^\s:]+)[^\n;]*?:=([^\n;]*)[\n;]", E)
        if m:
            V.declare([m.group(1)])
            V.link([m.group(1)] + sorted(V.mentioned(m.group(2))))
            return _parts(E[m.end():], V)
        return [E]
    for sep in ("↔", "→"):
        ps = _split_top(E, sep)
        if len(ps) > 1:
            if sep == "↔":
                return ps
            for prem in ps[:-1]:
                V.link(sorted(V.mentioned(prem)))
            return _parts(ps[-1], V)
    out = []
    for p in _split_top(E, "∨"):
        out += _split_top(p, "∧")
    return out


def split_statement(statement):
    """'theorem name (binders) : conclusion := by' -> (binders, conclusion), comments removed; None if it cannot split."""
    s = utils.rstrip_space_newline_and_by(utils.remove_comments(statement)).rstrip()
    s = s[:-2].rstrip() if s.endswith(":=") else s
    m = re.match(r"\s*theorem\s+\S+", s)
    s = s[m.end():] if m else s
    depth = 0
    for i, c in enumerate(s):
        depth += c in OPEN
        depth -= c in CLOSE
        if depth == 0 and c == ":" and not s.startswith(":=", i):
            return s[:i].strip(), s[i + 1:].strip()
    return None


def glued(statement):
    """Return (is_glued, parts, groups). groups lists part indices that share linked variables."""
    sp = split_statement(statement)
    if sp is None:
        return False, [statement], [[0]]
    binders, conclusion = sp
    V = _Vars()
    for g in _binder_groups(binders):
        _add_binder(V, g)
    parts = [p for p in _parts(conclusion, V) if p]
    if len(parts) < 2:
        return False, parts, [[0]]
    roots = [{V.find(v) for v in V.mentioned(p)} for p in parts]
    groups = []
    for i, r in enumerate(roots):
        hit = [g for g in groups if r & set().union(*(roots[j] for j in g))]
        merged = [i] + [j for g in hit for j in g]
        groups = [g for g in groups if g not in hit] + [sorted(merged)]
    return len(groups) > 1, parts, groups


def selftest():
    assert analysis_leak("<analysis>\n1. ...") and not analysis_leak("Prove x > 0") and not analysis_leak(None)
    # D1 pair_00012: inequality in a b c ∧ unrelated word problem in v
    s = """theorem thm_GAR (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) :
  (1/3) * (a + b + c)^2 ≥ a * Real.sqrt (b * c) + b * Real.sqrt (c * a) + c * Real.sqrt (a * b) ∧
  (∀ (v : ℝ) (hv : 0 < v),
    let d₁ := (100 : ℝ)
    -- comment
    (d₁ / v = 9) → (∃ (t : ℝ), t = 0.5)) := by"""
    assert glued(s)[0]
    # D1 pair_00002: trig inequality in K ↔ Bézout in a b
    s = """theorem thm_GAR (a b : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) (K : ℝ) (hK : K > 1) :
  let d := Int.gcd a b
  (∀ x : ℝ, 0 < x → x < Real.pi / K →
    Real.sin (K * x) / Real.sin x < K * Real.exp (-(K^2 - 1) / 6 * x^2)) ↔
  (∃ u v : ℤ, a * u + b * v = d) := by"""
    assert glued(s)[0]
    # Parts linked through a hypothesis, or a ∀ whose body holds the ∧: not glued
    assert not glued("theorem t (x y : ℕ) (h : x + y = 7) : x = 3 ∧ y = 4 := by")[0]
    assert not glued("theorem t : ∀ x : ℝ, 0 < x → x ^ 2 > 0 ∧ x + 1 > 1 := by")[0]
    assert not glued("theorem t (x : ℝ) (h : 0 < x) : x ^ 2 > 0 := by")[0]
    # Premise links the parts; a closed part is glued
    assert not glued("theorem t (x y : ℝ) : x < y → (x < 1 ∧ y > 0) := by")[0]
    assert glued("theorem t (x : ℝ) (h : 0 < x) : x ^ 2 > 0 ∧ (2 : ℕ) + 2 = 4 := by")[0]
    # ∧ under a premise stays split; ∧ inside an ∃ is one part
    assert glued("theorem t (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : x > 0 ∧ y > 0 := by")[0]
    assert not glued("theorem t (x : ℕ) : ∃ k : ℕ, x = 2 * k ∧ k > 0 := by")[0]
    print("filters selftest ok")


if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    selftest()
