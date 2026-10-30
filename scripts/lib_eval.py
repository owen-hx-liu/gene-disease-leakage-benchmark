#!/usr/bin/env python3
"""
lib_eval.py -- the single, shared evaluation harness for the leakage-aware
gene->disease link-prediction benchmark.

Both consumers import this module so every method is scored by byte-identical
code on byte-identical held-out edges:

    run_baselines.py  (Machine B)  -- Random / Common-Neighbors / Adamic-Adar
    run_kge.py        (Machine A)  -- TransE / DistMult / ComplEx / RotatE

There is exactly ONE owner of the ranking + metric protocol (this file) so the
numbers can never silently diverge between the two machines. See
WORK_SPLIT_3_MACHINES.md and PAPER_SCOPE.md.

--------------------------------------------------------------------------- #
Inputs (produced once by scripts/build_deleaked_splits.py, in data/processed/splits/)
--------------------------------------------------------------------------- #
    train.csv                        R0 training graph  (source_id,relation,target_id)
    valid.csv                        held-out target edges (KGE early stopping; not read here)
    test.csv                         ranking targets for every regime
    train_R1_redundancy.csv          R1 training graph (direct held-pair edges removed)
    train_R2_degree_null_seed42.csv  R2 training graph (degree-preserving null; built by Unit 7)
    train_R3_orthology_blocked.csv   R3 training graph (orthology/model-organism removed)
    hub_nodes.txt                    high-degree nodes (diagnostics; hub_filter is dormant)
    split_manifest.json              seed, counts, and the regime->file map (authoritative)

--------------------------------------------------------------------------- #
The four AUDIT regimes (the file->regime map is read from split_manifest.json)
--------------------------------------------------------------------------- #
    R0 Standard              train = train.csv                       test = test.csv
    R1 Redundancy-controlled train = train_R1_redundancy.csv         test = test.csv
    R2 Degree-controlled     train = train_R2_degree_null_seed42.csv test = test.csv   (Unit 7)
    R3 Orthology-blocked     train = train_R3_orthology_blocked.csv  test = test.csv

--------------------------------------------------------------------------- #
Ranking protocol  (reproduces the "sampled-50neg" rows in benchmark_results.json)
--------------------------------------------------------------------------- #
For each held-out test edge (gene g, true_disease d*):
    candidate set = { d* } + n_neg type-matched disease negatives
        * same category as d* (category_of == 'disease'),
        * excluding every disease already known to be associated with g in
          train U test (the *filtered* setting), and d* itself,
        * excluding hub nodes when hub_filter is True.
    score every candidate with score_fn(g, d); the rank of d* is
        rank = 1 + #(neg > true) + 0.5 * #(neg == true)      (realistic/tie-averaged)
No test edge is skipped: score_fn must return a finite float for every pair
(return a low sentinel, e.g. 0.0 or -1e30, for pairs it cannot score). Because
every method sees the identical candidate sets in the identical order, per-edge
reciprocal ranks are aligned across methods and can be paired-bootstrapped.

Hub-filter (legacy mechanism, DORMANT under the audit plan):
    All four audit regimes set hub_filter=False, so hub_set is always empty and the
    ranking is the plain filtered protocol above. The mechanism is retained (a True
    hub_filter removes hub nodes from the negative pool, and the caller may also drop
    them from topological evidence) in case a hub-controlled regime is re-added.

Deterministic: fixed seed (default 42), fixed test-edge order (file order).
Pure numpy/pandas -- no networkx / torch / sklearn, so it imports on every machine.

Self-test:
    python scripts/lib_eval.py                 # synthetic random-scorer control (fast)
    python scripts/lib_eval.py --regime R0     # same control on the real R0 split
A RANDOM scorer must land at chance (MRR ~ H_(n_neg+1)/(n_neg+1) ~ 0.089 for
n_neg=50; AUROC ~ 0.50). If it does not, the harness has a bug.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from kg_categories import category_of
except ImportError:  # allow running from the repo root
    from scripts.kg_categories import category_of

try:
    from config import PROCESSED_DIR
    DEFAULT_SPLITS_DIR = Path(PROCESSED_DIR) / "splits"
except Exception:
    DEFAULT_SPLITS_DIR = Path("data/processed/splits")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# --------------------------------------------------------------------------- #
# Metric primitives (no sklearn; tie-aware; unit-checked). Copied verbatim from
# kge_benchmark.py so classification numbers match byte-for-byte across scripts.
# --------------------------------------------------------------------------- #
def auroc(pos, neg) -> float:
    """Area under the ROC curve via the rank-sum (Mann-Whitney) identity."""
    pos = np.asarray(pos, dtype=float)
    neg = np.asarray(neg, dtype=float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    s = np.concatenate([pos, neg])
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    order = np.argsort(s, kind="stable")
    ranks = np.empty(len(s))
    sx = s[order]
    i = 0
    while i < len(s):  # average ranks within tied groups
        j = i
        while j + 1 < len(s) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    n_p, n_n = len(pos), len(neg)
    return float((ranks[y == 1].sum() - n_p * (n_p + 1) / 2.0) / (n_p * n_n))


def auprc(pos, neg) -> float:
    """Area under the precision-recall curve (interpolation-free step sum)."""
    pos = np.asarray(pos, dtype=float)
    neg = np.asarray(neg, dtype=float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    s = np.concatenate([pos, neg])
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    order = np.argsort(-s, kind="stable")
    y = y[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    prec = tp / (tp + fp)
    rec = tp / y.sum()
    rec = np.concatenate([[0], rec])
    prec = np.concatenate([[1], prec])
    return float(np.sum((rec[1:] - rec[:-1]) * prec[1:]))


# --------------------------------------------------------------------------- #
# Regime loading
# --------------------------------------------------------------------------- #
class Regime:
    """A loaded evaluation regime.

    Unpacks as the documented 3-tuple ``(train_edges, test_edges, hub_set)`` --
    ``train, test, hubs = load_regime("R0")`` -- while also exposing ``.hub_filter``,
    ``.name`` and the source filenames for provenance.

    * ``train_edges`` / ``test_edges``: numpy object arrays of shape (N, 3),
      columns [source_id, relation, target_id].
    * ``hub_set``: frozenset of hub node IDs *for this regime* -- the real hub set
      when hub_filter is True, an empty frozenset when it is False (so its truthiness
      already encodes the regime's hub policy).
    """

    __slots__ = ("name", "train_edges", "test_edges", "hub_set", "hub_filter",
                 "train_file", "test_file")

    def __init__(self, name, train_edges, test_edges, hub_set, hub_filter,
                 train_file, test_file):
        self.name = name
        self.train_edges = train_edges
        self.test_edges = test_edges
        self.hub_set = hub_set
        self.hub_filter = hub_filter
        self.train_file = train_file
        self.test_file = test_file

    def __iter__(self):
        return iter((self.train_edges, self.test_edges, self.hub_set))

    def __repr__(self):
        return (f"Regime({self.name}: train={self.train_file}[{len(self.train_edges):,}] "
                f"test={self.test_file}[{len(self.test_edges):,}] "
                f"hub_filter={self.hub_filter} hubs={len(self.hub_set):,})")


def _load_edges(path) -> np.ndarray:
    """Load a split CSV -> (N, 3) object array [source_id, relation, target_id]."""
    cols = ["source_id", "relation", "target_id"]
    df = pd.read_csv(path, usecols=cols, dtype=str)
    df = df.dropna(subset=cols)
    return df[cols].to_numpy(dtype=object)


def _load_hub_nodes(path) -> frozenset:
    """Load hub_nodes.txt (index column of node IDs, plus a 'degree' column)."""
    path = Path(path)
    if not path.exists():
        return frozenset()
    df = pd.read_csv(path)
    ids = df.iloc[:, 0].astype(str)
    ids = ids[(ids != "") & (ids.str.lower() != "nan")]
    return frozenset(ids.tolist())


def _regime_key(manifest: dict, regime: str) -> str:
    """Resolve 'R0' / 'r0' / 'R0_standard' to the manifest's regime key."""
    regimes = manifest["regimes"]
    if regime in regimes:
        return regime
    short = regime.split("_", 1)[0].upper()  # 'R0_standard' -> 'R0'
    by_short = {k.split("_", 1)[0].upper(): k for k in regimes}
    if short in by_short:
        return by_short[short]
    raise KeyError(f"unknown regime {regime!r}; options: "
                   f"{sorted(regimes) + sorted(by_short)}")


def load_regime(regime: str, splits_dir=None) -> Regime:
    """Load a regime's train/test edges and hub set per split_manifest.json.

    Parameters
    ----------
    regime : one of R0/R1/R2/R3 (short form or full manifest key).
    splits_dir : directory holding the split files + split_manifest.json.
                 Defaults to <repo>/data/processed/splits.

    Returns a :class:`Regime` that unpacks to ``(train_edges, test_edges, hub_set)``.
    """
    splits_dir = Path(splits_dir) if splits_dir is not None else DEFAULT_SPLITS_DIR
    manifest = json.loads((splits_dir / "split_manifest.json").read_text())
    key = _regime_key(manifest, regime)
    spec = manifest["regimes"][key]

    hub_filter = bool(spec.get("hub_filter", False))
    train_path = splits_dir / spec["train"]
    test_path = splits_dir / spec["test"]
    if not train_path.exists():
        raise FileNotFoundError(
            f"regime {key} needs '{spec['train']}' which does not exist yet. "
            + (spec.get("produced_by", "")
               or "run scripts/build_deleaked_splits.py first."))
    log(f"load_regime {key}: train={spec['train']} test={spec['test']} "
        f"hub_filter={hub_filter}")
    train_edges = _load_edges(train_path)
    test_edges = _load_edges(test_path)
    # Hub set is regime-specific: the real hubs only when the regime filters them.
    hub_set = _load_hub_nodes(splits_dir / "hub_nodes.txt") if hub_filter else frozenset()
    return Regime(key, train_edges, test_edges, hub_set, hub_filter,
                  spec["train"], spec["test"])


# --------------------------------------------------------------------------- #
# Ranking harness
# --------------------------------------------------------------------------- #
def _category_pools(train_edges: np.ndarray, hub_set) -> dict:
    """Map category -> unique node IDs of that category in the train graph.

    Hub nodes are removed so they can never be sampled as negatives (see
    hub-filter semantics in the module docstring). ``hub_set`` is empty for
    non-hub-filtered regimes, so this is a no-op there.
    """
    nodes = np.unique(np.concatenate([train_edges[:, 0], train_edges[:, 2]]))
    if hub_set:
        hs = np.fromiter((n not in hub_set for n in nodes), dtype=bool, count=len(nodes))
        nodes = nodes[hs]
    cats = np.array([category_of(n) for n in nodes], dtype=object)
    return {c: nodes[cats == c] for c in np.unique(cats)}


def _known_tails_by_head(train_edges, test_edges, heads_of_interest) -> dict:
    """head -> set of all tails linked to it in train U test (any relation).

    Only heads in ``heads_of_interest`` (the test genes) are tracked, so the
    dict stays small on the full 5.8M-edge graph.
    """
    hoi = set(heads_of_interest)
    known: dict = {}
    for arr in (train_edges, test_edges):
        h_col, t_col = arr[:, 0], arr[:, 2]
        mask = np.isin(h_col, list(hoi))
        for h, t in zip(h_col[mask], t_col[mask]):
            known.setdefault(h, set()).add(t)
    return known


def rank_test_edges(score_fn, train_edges, test_edges, hub_set, hub_filter,
                    n_neg=50, seed=42, return_scores=False, pools=None, known=None,
                    progress=None):
    """Rank each held-out true disease against n_neg type-matched negatives.

    Parameters
    ----------
    score_fn : callable ``score_fn(gene_id, disease_id) -> float``; higher = more
               likely a true edge. Must return a finite float for EVERY pair
               (use a low sentinel for pairs it cannot score) -- edges are never
               skipped, so all methods stay aligned edge-for-edge.
    train_edges, test_edges : (N, 3) object arrays from :func:`load_regime`.
    hub_set : frozenset of hub IDs (empty when not hub-filtering).
    hub_filter : bool -- remove hubs from the negative pool (topological evidence
                 hub-blocking is done inside score_fn by the caller).
    n_neg : negatives per test edge (default 50 -> 51-way ranking, chance MRR ~0.089).
    seed : RNG seed for negative sampling (default 42).
    progress : optional callable ``progress(done, total)`` invoked once per ranked
               test edge (plus a final call). Lets a caller draw a live progress bar
               over the otherwise-opaque ranking loop; it must self-throttle. None
               (default) keeps the loop silent -- backward compatible.

    Returns
    -------
    ranks : float array, one tie-averaged rank per test edge (in file order).
    """
    if not hub_filter:
        hub_set = frozenset()
    rng = np.random.default_rng(seed)

    # pools/known depend only on train/test edges (not the seed), so a caller running
    # many methods/seeds on one regime can build them once and pass them in.
    if pools is None:
        pools = _category_pools(train_edges, hub_set)
    if known is None:
        known = _known_tails_by_head(train_edges, test_edges, test_edges[:, 0])

    ranks = np.empty(len(test_edges), dtype=float)
    pos_out = np.empty(len(test_edges), dtype=float) if return_scores else None
    neg_out = [] if return_scores else None
    for i, (g, _r, d_true) in enumerate(test_edges):
        cat = category_of(d_true)
        pool = pools.get(cat)
        if pool is None or len(pool) == 0:
            pool = pools.get("disease", np.array([], dtype=object))
        exclude = set(known.get(g, ()))
        exclude.add(d_true)

        negs = _sample_negatives(pool, exclude, n_neg, rng)
        cand = [d_true] + list(negs)
        scores = np.fromiter((float(score_fn(g, d)) for d in cand),
                             dtype=float, count=len(cand))
        true_s, neg_s = scores[0], scores[1:]
        ranks[i] = 1.0 + float(np.sum(neg_s > true_s)) + 0.5 * float(np.sum(neg_s == true_s))
        if return_scores:
            pos_out[i] = true_s
            neg_out.append(neg_s)
        if progress is not None:
            progress(i + 1, len(test_edges))
    if progress is not None:
        progress(len(test_edges), len(test_edges))
    if return_scores:
        neg_cat = np.concatenate(neg_out) if neg_out else np.array([], dtype=float)
        return ranks, pos_out, neg_cat
    return ranks


def _sample_negatives(pool, exclude, n_neg, rng):
    """Draw up to n_neg IDs from ``pool`` excluding ``exclude`` (no replacement)."""
    if len(pool) == 0:
        return []
    # First try an oversampled single draw (cheap; almost always sufficient).
    take = min(len(pool), n_neg + len(exclude) + 16)
    draw = rng.choice(pool, size=take, replace=False)
    negs = [d for d in draw if d not in exclude][:n_neg]
    if len(negs) >= n_neg or take >= len(pool):
        return negs
    # Rare fallback: shuffle the whole pool and take what we can.
    perm = rng.permutation(pool)
    return [d for d in perm if d not in exclude][:n_neg]


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def ranking_metrics(ranks) -> dict:
    """ranks (from rank_test_edges) -> MRR, Hits@1/3/10, n."""
    ranks = np.asarray(ranks, dtype=float)
    n = len(ranks)
    if n == 0:
        return {"MRR": float("nan"), "Hits@1": float("nan"),
                "Hits@3": float("nan"), "Hits@10": float("nan"), "n": 0}
    return {
        "MRR": float(np.mean(1.0 / ranks)),
        "Hits@1": float(np.mean(ranks <= 1)),
        "Hits@3": float(np.mean(ranks <= 3)),
        "Hits@10": float(np.mean(ranks <= 10)),
        "n": int(n),
    }


def classification_metrics(pos_scores, neg_scores) -> dict:
    """Positive vs. negative scores -> AUROC, AUPRC.

    ``neg_scores`` may be random OR type-matched negatives; call once per negative
    mode (the current benchmark reports both) exactly as kge_benchmark.py does.
    """
    return {
        "AUROC": auroc(pos_scores, neg_scores),
        "AUPRC": auprc(pos_scores, neg_scores),
        "n_pos": int(len(pos_scores)),
        "n_neg": int(len(neg_scores)),
    }


def bootstrap_ci(values, n=1000, seed=42, alpha=0.05) -> dict:
    """Nonparametric bootstrap of the mean -> {mean, ci_low, ci_high, n}.

    Feed per-edge reciprocal ranks for an MRR CI, or per-edge 0/1 hit indicators
    for a Hits@k CI.
    """
    values = np.asarray(values, dtype=float)
    m = len(values)
    if m == 0:
        return {"mean": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, m, size=(n, m))
    boot = values[idx].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci_low": float(np.quantile(boot, alpha / 2)),
        "ci_high": float(np.quantile(boot, 1 - alpha / 2)),
        "n": int(m),
    }


def paired_bootstrap_pvalue(rr_method_a, rr_method_b, n=1000, seed=42,
                            alpha=0.05, chunk=2000) -> dict:
    """Paired bootstrap on per-edge reciprocal ranks (A - B).

    ``rr_method_a`` / ``rr_method_b`` are aligned per-edge reciprocal-rank arrays
    (i.e. ``1 / rank_test_edges(...)`` for two methods on the SAME regime, so
    element i is the same held-out edge). Returns the observed MRR difference, a
    95% CI on that difference, and a two-sided bootstrap p-value for H0: delta = 0.

    Resamples are drawn ``chunk`` at a time so that large ``n`` (e.g. 100,000 for
    multiple-testing correction) does not build an n x m index matrix in memory.
    ``p_value`` is the raw achieved significance level and is 0 when no resample
    reaches zero; ``p_value_plus_one`` adds one to each tail count, so it never
    falls below 2/(n+1) and is the value to correct for multiple testing.
    """
    a = np.asarray(rr_method_a, dtype=float)
    b = np.asarray(rr_method_b, dtype=float)
    if len(a) != len(b):
        raise ValueError(f"paired arrays must align: {len(a)} != {len(b)}")
    d = a - b
    m = len(d)
    if m == 0:
        return {"delta": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "p_value": float("nan"),
                "p_value_plus_one": float("nan"), "n": 0, "n_boot": int(n)}
    obs = float(d.mean())
    rng = np.random.default_rng(seed)
    boot = np.empty(n, dtype=float)
    for s in range(0, n, chunk):
        k = min(chunk, n - s)
        boot[s:s + k] = d[rng.integers(0, m, size=(k, m))].mean(axis=1)
    # Two-sided achieved significance level from the bootstrap distribution.
    le0, ge0 = int(np.sum(boot <= 0)), int(np.sum(boot >= 0))
    p = 2.0 * min(le0, ge0) / n
    p1 = 2.0 * (min(le0, ge0) + 1) / (n + 1)
    return {
        "delta": obs,
        "ci_low": float(np.quantile(boot, alpha / 2)),
        "ci_high": float(np.quantile(boot, 1 - alpha / 2)),
        "p_value": min(1.0, p),
        "p_value_plus_one": min(1.0, p1),
        "n": int(m),
        "n_boot": int(n),
    }


# --------------------------------------------------------------------------- #
# Self-test: a RANDOM scorer must sit at chance. If not, the harness is buggy.
# --------------------------------------------------------------------------- #
def _synthetic_split(seed=0, n_gene=200, n_dis=800, n_train=4000, n_test=300):
    """Small synthetic gene->disease graph for a fast, file-free self-test."""
    rng = np.random.default_rng(seed)
    genes = np.array([f"HGNC:{i}" for i in range(n_gene)], dtype=object)
    dis = np.array([f"MONDO:{i:07d}" for i in range(n_dis)], dtype=object)
    rel = "BIOLINK:GENE_ASSOCIATED_WITH_CONDITION"

    def rand_pairs(k):
        g = genes[rng.integers(0, n_gene, k)]
        d = dis[rng.integers(0, n_dis, k)]
        return set(zip(g.tolist(), d.tolist()))

    train_pairs = rand_pairs(n_train)
    test_pairs = list(rand_pairs(n_test * 2) - train_pairs)[:n_test]
    train = np.array([[g, rel, d] for g, d in train_pairs], dtype=object)
    test = np.array([[g, rel, d] for g, d in test_pairs], dtype=object)
    return train, test


def _self_test(regime=None, splits_dir=None, n_neg=50):
    if regime:
        train, test, hubs = load_regime(regime, splits_dir)
        hub_filter = bool(hubs)
        log(f"self-test on real split {regime}: "
            f"train={len(train):,} test={len(test):,} hubs={len(hubs):,}")
    else:
        train, test = _synthetic_split()
        hubs, hub_filter = frozenset(), False
        log(f"self-test on synthetic split: train={len(train):,} test={len(test):,}")

    score_rng = np.random.default_rng(123)
    random_scorer = lambda g, d: float(score_rng.random())  # noqa: E731

    ranks = rank_test_edges(random_scorer, train, test, hubs, hub_filter,
                            n_neg=n_neg, seed=42)
    rm = ranking_metrics(ranks)
    chance = float(np.sum(1.0 / np.arange(1, n_neg + 2)) / (n_neg + 1))  # H_(k)/k
    rr = 1.0 / ranks
    ci = bootstrap_ci(rr)

    cls_rng = np.random.default_rng(7)
    pos = cls_rng.random(2000)
    neg = cls_rng.random(2000)
    cm = classification_metrics(pos, neg)

    # A method paired against itself: zero difference, non-significant.
    self_paired = paired_bootstrap_pvalue(rr, rr)

    log(f"random-scorer ranking: MRR={rm['MRR']:.4f} (chance~{chance:.4f}) "
        f"Hits@1={rm['Hits@1']:.4f} Hits@10={rm['Hits@10']:.4f} "
        f"MRR 95% CI=[{ci['ci_low']:.4f},{ci['ci_high']:.4f}]")
    log(f"random-scorer classification: AUROC={cm['AUROC']:.4f} AUPRC={cm['AUPRC']:.4f}")
    log(f"self-paired bootstrap: delta={self_paired['delta']:.4f} "
        f"p={self_paired['p_value']:.3f}")

    # Assertions: the negative controls MUST pass.
    tol_mrr = 0.04
    assert abs(rm["MRR"] - chance) < tol_mrr, \
        f"MRR {rm['MRR']:.4f} not near chance {chance:.4f} -- harness bug"
    assert 0.44 < cm["AUROC"] < 0.56, f"AUROC {cm['AUROC']:.4f} not ~0.5 -- harness bug"
    assert 0.40 < cm["AUPRC"] < 0.60, f"AUPRC {cm['AUPRC']:.4f} not ~0.5 -- harness bug"
    assert abs(self_paired["delta"]) < 1e-9, "A-minus-A delta must be exactly 0"
    log("SELF-TEST PASSED: negative controls at chance.")


def main():
    ap = argparse.ArgumentParser(description="lib_eval self-test / negative-control check")
    ap.add_argument("--regime", default=None,
                    help="run the random-scorer control on a real split (R0/R1/R2/R3); "
                         "default is a fast synthetic split")
    ap.add_argument("--splits-dir", default=None)
    ap.add_argument("--n-neg", type=int, default=50)
    args = ap.parse_args()
    _self_test(args.regime, args.splits_dir, args.n_neg)


if __name__ == "__main__":
    main()
