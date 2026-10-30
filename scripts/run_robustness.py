#!/usr/bin/env python3
"""
run_robustness.py -- full-ranking robustness check for the leakage audit.

Why this exists
---------------
Table 2 (run_baselines.py / run_kge.py) ranks each held-out true disease against
only **50 type-matched sampled negatives** (lib_eval.rank_test_edges, n_neg=50).
That is standard and keeps KGE tractable, but a reviewer's first objection is:
"sampled-50-neg MRR is optimistic; the true test is ranking against *all*
candidate entities." This script answers that objection for the topological
baselines -- the methods that carry the degree story -- by re-ranking the true
disease against the **entire filtered disease pool** (D = all disease-category
nodes in the training graph, ~14.3k), under the *same* filtering as lib_eval
(exclude every disease already linked to the gene in train U test, and the true
disease itself). Same scorers, same graph, strictly harder ranking.

The question it settles: does the degree pattern survive full ranking?
  * PreferentialAttachment (pure degree) should stay the strongest baseline.
  * The overlap heuristics (CommonNeighbors / AdamicAdar / Jaccard) should still
    drop sharply from R0 (standard) to R2 (degree-preserving null).
If yes, the centerpiece decomposition is not an artifact of the 50-negative
protocol. (KGE full-ranking needs the trained embeddings, which are not on disk
here; it is a noted GPU-env follow-up, see the report.)

Correctness
-----------
Scores come from the byte-identical scorers in run_baselines.make_scorers
(hub_cap=2000 for CN/AA, deg>1 for AA, Jaccard uncapped). Ranks use the exact
tie-averaged rule from lib_eval:  rank = 1 + #(neg>true) + 0.5*#(neg==true).
The fast per-method ranking is validated against a brute-force reference that
loops the run_baselines scorer over the full pool, on a --validate sample of
test edges; the run aborts if any rank disagrees.

Regimes: R0 and R2 by default (the pair the robustness claim is about).

Out: data/processed/results/robustness/robustness_<regime>.json  (one per regime).

Usage:
    python scripts/run_robustness.py                       # R0,R2 ; validate 150 edges
    python scripts/run_robustness.py --regimes R0,R1,R2,R3 # all four
    python scripts/run_robustness.py --validate 0          # skip brute-force check
    python scripts/run_robustness.py --limit 300           # quick smoke on 300 edges
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

try:
    import lib_eval
    import run_baselines
    from kg_categories import category_of
except ImportError:  # allow running from repo root
    from scripts import lib_eval, run_baselines
    from scripts.kg_categories import category_of

try:
    from config import PROCESSED_DIR
    DEFAULT_OUT = Path(PROCESSED_DIR) / "results" / "robustness"
except Exception:
    DEFAULT_OUT = Path("data/processed/results/robustness")

METHODS = ["Random", "CommonNeighbors", "AdamicAdar", "Jaccard",
           "PreferentialAttachment"]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# --------------------------------------------------------------------------- #
# Pool construction
# --------------------------------------------------------------------------- #
def build_disease_pool(adj, deg):
    """All disease-category nodes present in the training graph -> (ids, degs).

    This is the full candidate set for ranking (the 'filtered full ranking'
    denominator). ids is an object array; degs the aligned int degree array.
    """
    ids = np.array([n for n in adj.keys() if category_of(n) == "disease"],
                   dtype=object)
    degs = np.array([deg[n] for n in ids], dtype=np.int64)
    return ids, degs


def known_tails_by_head_fast(train_edges, test_edges, heads_of_interest):
    """head -> set of tails linked to it in train U test (same semantics as
    lib_eval._known_tails_by_head, but a single Python pass instead of np.isin,
    which is pathologically slow on 5.8M object rows in numpy 2.x)."""
    hoi = set(heads_of_interest.tolist())
    known: dict = {}
    for arr in (train_edges, test_edges):
        src = arr[:, 0].tolist()
        tgt = arr[:, 2].tolist()
        for h, t in zip(src, tgt):
            if h in hoi:
                s = known.get(h)
                if s is None:
                    known[h] = s = set()
                s.add(t)
    return known


def disease_neighbors(adj, dis_set, cache, w):
    """Disease-category neighbors of node w (cached), as a list of ids."""
    dn = cache.get(w)
    if dn is None:
        dn = [x for x in adj[w] if x in dis_set]
        cache[w] = dn
    return dn


# --------------------------------------------------------------------------- #
# Fast exact full-ranking, one method family at a time
# --------------------------------------------------------------------------- #
def rank_all_full(adj, deg, dis_ids, dis_degs, dis_set, test_edges, known,
                  hub_cap, rand_seed, progress=None, limit=None):
    """Full-pool tie-averaged rank of each true disease, per method.

    Returns dict method -> np.ndarray of ranks (len = #ranked edges).
    Ranks are exact and identical to looping the run_baselines scorer over the
    entire filtered pool (verified by validate_against_bruteforce).
    """
    D = len(dis_ids)
    # PrefAttach: pre-sort pool degrees for O(log D) counting per edge.
    order = np.argsort(dis_degs, kind="stable")
    degs_sorted = dis_degs[order]
    # map disease id -> its degree (for excluded-set corrections)
    # (deg already has it, but keep explicit for clarity)

    n_edges = len(test_edges) if limit is None else min(limit, len(test_edges))
    out = {m: np.empty(n_edges, dtype=float) for m in METHODS}
    log_ = math.log
    dn_cache = {}
    rng = np.random.default_rng(rand_seed + 10_000)  # matches run_baselines Random seed

    for i in range(n_edges):
        g, _r, d_true = test_edges[i]
        Ng = adj.get(g)
        degg = deg.get(g, 0)

        # filtered negatives = pool diseases minus known tails of g minus d_true
        excl = set(known.get(g, ()))
        excl.add(d_true)
        excl_dis = [d for d in excl if d in dis_set]      # only diseases shrink the pool
        n_excl_in_pool = len(excl_dis)
        n_filt = D - n_excl_in_pool                       # size of the negative pool

        # ---- Random: analytic chance rank over a uniform ranking of n_filt+1 slots
        # E[1/rank] for the true edge placed uniformly among (n_filt+1) items is
        # H_{n_filt+1}/(n_filt+1); we store that expected reciprocal rank directly
        # by storing an "effective rank" = 1 / that expectation is not linear, so
        # instead store the exact per-edge expected RR in a parallel array.
        # (handled after the loop via _random_expected_rr using n_filt_arr)
        # placeholder; real value filled below
        # ---- gather reachable disease scores for overlap methods -------------
        cn = {}   # capped common-neighbor count      (deg(w) <= hub_cap)
        aa = {}   # adamic-adar sum   (1 < deg(w) <= hub_cap)
        jn = {}   # uncapped common-neighbor count     (jaccard numerator)
        if Ng:
            for w in Ng:
                dw = deg[w]
                capped = dw <= hub_cap
                aaw = (1.0 / log_(dw)) if (1 < dw <= hub_cap) else 0.0
                for x in disease_neighbors(adj, dis_set, dn_cache, w):
                    jn[x] = jn.get(x, 0) + 1
                    if capped:
                        cn[x] = cn.get(x, 0) + 1
                        if aaw:
                            aa[x] = aa.get(x, 0.0) + aaw

        # ---- CommonNeighbors -------------------------------------------------
        out["CommonNeighbors"][i] = _rank_from_dict(
            cn, d_true, excl, n_filt, score_true=cn.get(d_true, 0.0))
        # ---- AdamicAdar ------------------------------------------------------
        out["AdamicAdar"][i] = _rank_from_dict(
            aa, d_true, excl, n_filt, score_true=aa.get(d_true, 0.0))
        # ---- Jaccard: score = num / (deg(g)+deg(d)-num) ----------------------
        jac = {}
        for d, num in jn.items():
            denom = degg + deg[d] - num
            jac[d] = (num / denom) if denom > 0 else 0.0
        out["Jaccard"][i] = _rank_from_dict(
            jac, d_true, excl, n_filt, score_true=jac.get(d_true, 0.0))
        # ---- PreferentialAttachment: rank by deg(d) --------------------------
        out["PreferentialAttachment"][i] = _rank_pref(
            degg, deg.get(d_true, 0), d_true, excl_dis, deg,
            degs_sorted, n_filt)
        # ---- Random (store n_filt now; convert to expected RR after loop) ----
        out["Random"][i] = n_filt   # temporarily hold n_filt; see below

        if progress is not None:
            progress(i + 1, n_edges)
    if progress is not None:
        progress(n_edges, n_edges)

    # Random: replace stored n_filt with an *effective rank* whose reciprocal is
    # the exact expected reciprocal rank H_{n+1}/(n+1). We store 1/E[RR] so that
    # ranking_metrics(1/rank)=E[RR]; but Hits@k for Random are handled separately
    # in summarize_random (a single scalar rank can't encode Hits@k). We keep the
    # ranks array meaningful for MRR and compute Random Hits@k analytically later.
    nfilt_arr = out["Random"].astype(np.int64)
    out["Random"] = _random_effective_rank(nfilt_arr)
    out["_n_filt"] = nfilt_arr  # exposed for reporting + Random Hits@k
    return out


def _rank_from_dict(score_map, d_true, excl, n_filt, score_true):
    """Tie-averaged full-pool rank of d_true given a dict of nonzero neg scores.

    score_map: disease -> score, containing ONLY diseases with score > 0
               (reachable via a shared neighbor). Every other filtered negative
               scores exactly 0. excl is the excluded set (known tails + d_true).
    """
    n_gt = 0
    n_eq = 0
    n_pos = 0   # filtered negatives with score > 0
    for d, s in score_map.items():
        if d in excl:          # d_true and known tails are not negatives
            continue
        n_pos += 1
        if s > score_true:
            n_gt += 1
        elif s == score_true:
            n_eq += 1
    if score_true > 0:
        return 1.0 + n_gt + 0.5 * n_eq
    # true scores 0: it ties with every zero-scoring filtered negative
    n_zero_neg = n_filt - n_pos
    return 1.0 + n_pos + 0.5 * n_zero_neg


def _rank_pref(degg, deg_true, d_true, excl_dis, deg, degs_sorted, n_filt):
    """Tie-averaged full-pool rank for PreferentialAttachment (score=deg(g)*deg(d))."""
    if degg == 0:
        # all scores are 0; true ties with the whole pool
        return 1.0 + 0.5 * n_filt
    # ranking by deg(d); count pool diseases with degree >/== deg_true
    n_gt_pool = int(len(degs_sorted) - np.searchsorted(degs_sorted, deg_true, side="right"))
    n_eq_pool = int(np.searchsorted(degs_sorted, deg_true, side="right")
                    - np.searchsorted(degs_sorted, deg_true, side="left"))
    # remove excluded pool diseases (known tails that are diseases, and d_true)
    n_gt_excl = 0
    n_eq_excl = 0
    for d in excl_dis:
        dd = deg[d]
        if dd > deg_true:
            n_gt_excl += 1
        elif dd == deg_true:
            n_eq_excl += 1
    n_gt = n_gt_pool - n_gt_excl
    n_eq = n_eq_pool - n_eq_excl        # n_eq_excl includes d_true itself if pooled
    return 1.0 + n_gt + 0.5 * max(n_eq, 0)


def _random_effective_rank(nfilt_arr):
    """Effective rank r s.t. 1/r == E[RR] == H_{n+1}/(n+1) for each n_filt."""
    eff = np.empty(len(nfilt_arr), dtype=float)
    for i, n in enumerate(nfilt_arr):
        k = int(n) + 1                       # candidates incl. the true edge
        H = np.log(k) + 0.5772156649015329 + 1.0 / (2 * k)  # harmonic approx
        # exact for small k
        if k <= 2000:
            H = float(np.sum(1.0 / np.arange(1, k + 1)))
        eff[i] = 1.0 / (H / k)
    return eff


# --------------------------------------------------------------------------- #
# Brute-force validation (loops the exact run_baselines scorer over full pool)
# --------------------------------------------------------------------------- #
def validate_against_bruteforce(adj, deg, dis_ids, dis_set, test_edges, known,
                                hub_cap, seed, n_sample, fast_ranks):
    """Assert fast ranks == brute-force ranks on n_sample random test edges.

    Brute force = run_baselines.make_scorers over the *entire* filtered pool,
    with the exact lib_eval tie rule. Covers the four deterministic methods
    (Random excluded -- it is analytic).
    """
    if n_sample <= 0:
        log("validation SKIPPED (--validate 0)")
        return
    scorers = run_baselines.make_scorers(adj, deg, hub_cap, seed)
    rng = np.random.default_rng(12345)
    idx = rng.choice(len(fast_ranks["CommonNeighbors"]),
                     size=min(n_sample, len(fast_ranks["CommonNeighbors"])),
                     replace=False)
    check_methods = ["CommonNeighbors", "AdamicAdar", "Jaccard",
                     "PreferentialAttachment"]
    mism = 0
    for j in idx:
        g, _r, d_true = test_edges[j]
        excl = set(known.get(g, ()))
        excl.add(d_true)
        negs = [d for d in dis_ids if d not in excl]
        for m in check_methods:
            fn = scorers[m]
            ts = float(fn(g, d_true))
            gt = eq = 0
            for d in negs:
                s = float(fn(g, d))
                if s > ts:
                    gt += 1
                elif s == ts:
                    eq += 1
            brute = 1.0 + gt + 0.5 * eq
            fast = fast_ranks[m][j]
            if abs(brute - fast) > 1e-6:
                mism += 1
                log(f"  MISMATCH {m} edge#{j} ({g}->{d_true}): "
                    f"brute={brute} fast={fast}")
    if mism:
        raise AssertionError(f"validation FAILED: {mism} rank mismatches")
    log(f"validation PASSED: {len(idx)} edges x {len(check_methods)} methods "
        f"agree exactly with brute force.")


# --------------------------------------------------------------------------- #
# Metrics / summary
# --------------------------------------------------------------------------- #
def summarize(ranks, n_filt):
    """MRR / Hits@k / mean-rank / median-rank from full-pool ranks."""
    ranks = np.asarray(ranks, dtype=float)
    return {
        "MRR": float(np.mean(1.0 / ranks)),
        "Hits@1": float(np.mean(ranks <= 1)),
        "Hits@3": float(np.mean(ranks <= 3)),
        "Hits@10": float(np.mean(ranks <= 10)),
        "Hits@100": float(np.mean(ranks <= 100)),
        "MeanRank": float(np.mean(ranks)),
        "MedianRank": float(np.median(ranks)),
        "n": int(len(ranks)),
    }


def summarize_random(n_filt):
    """Analytic Random full-ranking metrics (uniform placement of the true edge)."""
    n = np.asarray(n_filt, dtype=float)
    k = n + 1.0
    # exact expected MRR = mean H_{k}/k
    mrr = float(np.mean([np.sum(1.0 / np.arange(1, int(kk) + 1)) / kk for kk in k]))
    return {
        "MRR": mrr,
        "Hits@1": float(np.mean(1.0 / k)),
        "Hits@3": float(np.mean(np.minimum(3.0, k) / k)),
        "Hits@10": float(np.mean(np.minimum(10.0, k) / k)),
        "Hits@100": float(np.mean(np.minimum(100.0, k) / k)),
        "MeanRank": float(np.mean((k + 1.0) / 2.0)),
        "MedianRank": float(np.median((k + 1.0) / 2.0)),
        "n": int(len(n)),
        "note": "analytic chance floor (true edge placed uniformly at random)",
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--regimes", default="R0,R2")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random-scorer seed (only affects the Random control)")
    ap.add_argument("--hub-cap", type=int, default=2000)
    ap.add_argument("--validate", type=int, default=150,
                    help="brute-force-check this many test edges (0 to skip)")
    ap.add_argument("--limit", type=int, default=None,
                    help="rank only the first N test edges (smoke test)")
    ap.add_argument("--splits-dir", default=None)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    regimes = [r.strip() for r in args.regimes.split(",") if r.strip()]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for regime in regimes:
        log(f"================= regime {regime} (full ranking) =================")
        try:
            reg = lib_eval.load_regime(regime, args.splits_dir)
        except FileNotFoundError as e:
            log(f"SKIP {regime}: {e}")
            continue
        train, test, hubs = reg
        t0 = time.time()
        adj, deg = run_baselines.build_adjacency(
            train, progress=run_baselines.make_progress(f"{regime} adjacency"))
        log(f"  {len(adj):,} nodes in adjacency ({time.time() - t0:.1f}s)")
        t0 = time.time()
        dis_ids, dis_degs = build_disease_pool(adj, deg)
        dis_set = set(dis_ids.tolist())
        known = known_tails_by_head_fast(train, test, test[:, 0])
        log(f"  disease pool D={len(dis_ids):,}; built known-tails "
            f"({time.time() - t0:.1f}s)")

        t0 = time.time()
        ranks = rank_all_full(
            adj, deg, dis_ids, dis_degs, dis_set, test, known,
            args.hub_cap, args.seed,
            progress=run_baselines.make_progress(f"{regime} full-rank"),
            limit=args.limit)
        n_filt = ranks.pop("_n_filt")
        log(f"  ranked {len(n_filt):,} edges over full pool "
            f"({time.time() - t0:.1f}s)")
        log(f"  filtered pool size: mean={n_filt.mean():.0f} "
            f"min={n_filt.min()} max={n_filt.max()}")

        validate_against_bruteforce(
            adj, deg, dis_ids, dis_set, test, known, args.hub_cap, args.seed,
            args.validate, ranks)

        summary = {}
        for m in METHODS:
            summary[m] = (summarize_random(n_filt) if m == "Random"
                          else summarize(ranks[m], n_filt))

        payload = {
            "regime": regime,
            "regime_key": reg.name,
            "train_file": reg.train_file,
            "test_file": reg.test_file,
            "protocol": "full filtered ranking over ALL disease-category nodes",
            "pool_size_D": int(len(dis_ids)),
            "n_filt_mean": float(n_filt.mean()),
            "n_test_edges": int(len(n_filt)),
            "hub_cap": args.hub_cap,
            "seed": args.seed,
            "validated_edges": int(args.validate),
            "methods": summary,
            # aligned with test.csv row order, so regimes and methods can be paired
            # (multiple_testing.py). Random is omitted: its full-pool rank is analytic.
            "per_edge_reciprocal_ranks": {
                m: (1.0 / np.asarray(ranks[m], dtype=float)).tolist()
                for m in METHODS if m != "Random"},
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        p = out / f"robustness_{regime}.json"
        p.write_text(json.dumps(payload, indent=2))
        log(f"wrote {p}")
        log(f"--- {regime} full-ranking summary ---")
        for m in METHODS:
            s = summary[m]
            log(f"    {m:23s} MRR={s['MRR']:.4f}  H@10={s['Hits@10']:.4f}  "
                f"H@100={s['Hits@100']:.4f}  MedRank={s['MedianRank']:.0f}")
    log("DONE.")


if __name__ == "__main__":
    main()
