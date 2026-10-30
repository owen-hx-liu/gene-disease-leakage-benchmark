#!/usr/bin/env python3
"""multiple_testing.py -- paired bootstrap tests with Holm correction (S6 Table).

Recomputes every comparison in the pre-defined testing family from the stored
per-edge reciprocal ranks (no model is retrained, no graph is rescored) and
applies the Holm step-down correction over that family.

Why 100,000 resamples: a two-sided bootstrap p-value cannot fall below 2/(n+1),
and Holm multiplies the smallest p-value by the family size. With n = 1,000 the
floor is 0.002, so a family of more than 24 comparisons can never reach
alpha = 0.05 after correction however large the effect. With n = 100,000 the
floor is 2e-5.

Family (alpha = 0.05, both protocols):
  A. regime effects   m_X(R) vs m_X(R0), R in {R1, R2, R3}, the 8 non-random methods
  B. degree-only gap  m_X(R2) vs m_PA(R2), the 7 methods other than PA and Random
  C. method contrasts m_AA(R0) vs m_X(R0), the 7 non-random methods other than AA
The permutation test (Table 4) and the Hetionet replication are not in the family.

Per-edge inputs (all aligned with test.csv row order, 4,228 edges):
  sampled, baselines : results/baselines/baselines_R*.json, seed-42 negatives
  sampled, KGE       : results/kge/kge_<model>_<R>_seed*.json, averaged over seeds
  full,    baselines : results/robustness/robustness_R*.json
  full,    KGE       : results/kge_fullrank/kge_<model>_<R>_seed*.json, averaged over seeds

Also reports bootstrap CIs on the retained share m(R2)/m(R0), with numerator and
denominator computed on the same resampled edges.

Out: results/stats/multiple_testing.json, tables/tableS6_pvalues.{md,tex}

Usage:
    python scripts/multiple_testing.py                 # n = 100,000
    python scripts/multiple_testing.py --n-boot 1000   # quick look
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib_eval as le  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "processed" / "results"
TABLES = ROOT / "tables"

KGE = ["TransE", "RotatE", "DistMult", "ComplEx"]
BASE = ["CommonNeighbors", "AdamicAdar", "Jaccard", "PreferentialAttachment"]
METHODS = KGE + BASE
REGIMES = ["R0", "R1", "R2", "R3"]
PROTOCOLS = ["sampled", "full"]
LABEL = {"CommonNeighbors": "Common Neighbors", "AdamicAdar": "Adamic-Adar",
         "PreferentialAttachment": "PA", "Jaccard": "Jaccard"}
KGE_RE = re.compile(r"kge_(TransE|RotatE|DistMult|ComplEx)_(R[0-3])_seed(\d+)\.json$")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_kge(results_dir):
    """(model, regime) -> per-edge RR averaged over training seeds."""
    per = defaultdict(dict)
    for p in sorted(glob.glob(str(results_dir / "kge_*_seed*.json"))):
        mt = KGE_RE.search(p.replace("\\", "/"))
        if not mt:
            continue
        model, regime, seed = mt.group(1), mt.group(2), int(mt.group(3))
        per[(model, regime)][seed] = np.asarray(
            json.loads(Path(p).read_text())["reciprocal_ranks"], float)
    out = {}
    for key, seeds in per.items():
        mats = [seeds[s] for s in sorted(seeds)]
        if len({len(x) for x in mats}) != 1:
            raise SystemExit(f"{key}: seeds disagree on edge count; cannot pair")
        out[key] = (np.mean(np.vstack(mats), axis=0), sorted(seeds))
    return out


def load_baselines_sampled():
    out = {}
    for r in REGIMES:
        j = json.loads((RESULTS / "baselines" / f"baselines_{r}.json").read_text())
        i42 = j["per_edge_rr_seed_order"].index(42)
        for m in BASE:
            out[(m, r)] = (np.asarray(j["per_edge_reciprocal_ranks"][m][i42], float), [42])
    return out


def load_baselines_full():
    out = {}
    for r in REGIMES:
        j = json.loads((RESULTS / "robustness" / f"robustness_{r}.json").read_text())
        if "per_edge_reciprocal_ranks" not in j:
            raise SystemExit(f"robustness_{r}.json has no per-edge ranks; rerun "
                             "scripts/run_robustness.py --regimes R0,R1,R2,R3")
        for m in BASE:
            out[(m, r)] = (np.asarray(j["per_edge_reciprocal_ranks"][m], float), ["det"])
    return out


def holm(pvals):
    """Holm step-down adjusted p-values (same order as input)."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    k = len(p)
    adj = np.empty(k)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (k - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def ratio_ci(num, den, n, seed=42, alpha=0.05, chunk=2000):
    """Bootstrap CI of mean(num)/mean(den) on shared edge resamples."""
    m = len(num)
    rng = np.random.default_rng(seed)
    boot = np.empty(n)
    for s in range(0, n, chunk):
        k = min(chunk, n - s)
        idx = rng.integers(0, m, size=(k, m))
        boot[s:s + k] = num[idx].mean(axis=1) / den[idx].mean(axis=1)
    return (float(num.mean() / den.mean()), float(np.quantile(boot, alpha / 2)),
            float(np.quantile(boot, 1 - alpha / 2)))


def fmt_p(p):
    if p >= 0.001:
        return f"{p:.3f}"
    mant, exp = f"{p:.1e}".split("e")
    return f"{mant}e{int(exp)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=100_000)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()
    n = args.n_boot

    rr = {"sampled": {**load_baselines_sampled(), **load_kge(RESULTS / "kge")},
          "full": {**load_baselines_full(), **load_kge(RESULTS / "kge_fullrank")}}
    for P in PROTOCOLS:
        missing = [(m, r) for m in METHODS for r in REGIMES if (m, r) not in rr[P]]
        if missing:
            raise SystemExit(f"{P}: missing per-edge ranks for {missing}")
        lens = {len(v[0]) for v in rr[P].values()}
        if len(lens) != 1:
            raise SystemExit(f"{P}: per-edge vectors differ in length {lens}")

    tests = []  # (block, protocol, label, a, b) ; delta = mean(a - b)
    for P in PROTOCOLS:
        for X in METHODS:
            for R in ["R1", "R2", "R3"]:
                tests.append(("A regime effect", P, f"{LABEL.get(X, X)}: {R} vs R0",
                              rr[P][(X, R)][0], rr[P][(X, "R0")][0]))
    for P in PROTOCOLS:
        for X in METHODS:
            if X != "PreferentialAttachment":
                tests.append(("B R2 vs PA", P, f"{LABEL.get(X, X)} vs PA at R2",
                              rr[P][(X, "R2")][0], rr[P][("PreferentialAttachment", "R2")][0]))
    for P in PROTOCOLS:
        for X in METHODS:
            if X != "AdamicAdar":
                tests.append(("C method contrast", P, f"Adamic-Adar vs {LABEL.get(X, X)} at R0",
                              rr[P][("AdamicAdar", "R0")][0], rr[P][(X, "R0")][0]))

    log(f"family size m = {len(tests)}; {n:,} resamples per test")
    rows = []
    t0 = time.time()
    for i, (block, P, label, a, b) in enumerate(tests, 1):
        s = le.paired_bootstrap_pvalue(a, b, n=n)
        rows.append(dict(block=block, protocol=P, comparison=label,
                         mrr_a=float(a.mean()), mrr_b=float(b.mean()),
                         delta=s["delta"], ci_low=s["ci_low"], ci_high=s["ci_high"],
                         p_raw=s["p_value"], p=s["p_value_plus_one"], n_edges=s["n"]))
        if i % 10 == 0:
            log(f"  {i}/{len(tests)} tests ({time.time() - t0:.0f}s)")
    adj = holm([r["p"] for r in rows])
    for r, pa in zip(rows, adj):
        r["p_holm"] = float(pa)
        r["significant"] = bool(pa < args.alpha)

    retained = []
    for P in PROTOCOLS:
        for X in METHODS:
            est, lo, hi = ratio_ci(rr[P][(X, "R2")][0], rr[P][(X, "R0")][0], n)
            retained.append(dict(protocol=P, method=X, retained_share=est,
                                 ci_low=lo, ci_high=hi))

    out_dir = RESULTS / "stats"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(
        description="Paired bootstrap tests over the pre-defined family, Holm-corrected.",
        n_boot=n, alpha=args.alpha, family_size=len(rows), seed=42,
        p_value_definition="two-sided, 2*(min tail count + 1)/(n_boot + 1)",
        kge_seed_handling="per-edge reciprocal ranks averaged over training seeds",
        baseline_sampled_negatives="seed 42",
        tests=rows, retained_share=retained,
        created=time.strftime("%Y-%m-%d %H:%M:%S"))
    (out_dir / "multiple_testing.json").write_text(json.dumps(payload, indent=2))

    TABLES.mkdir(exist_ok=True)
    md = ["### S6 Table. Paired bootstrap tests with Holm correction", "",
          f"*Two-sided paired bootstrap on the 4,228 per-edge reciprocal ranks, {n:,} "
          f"resamples, seed 42; Holm correction over all {len(rows)} comparisons "
          f"(alpha = {args.alpha}). Delta = MRR of the first term minus the second. "
          "KGE per-edge ranks are averaged over seeds 42, 1 and 7; sampled-ranking "
          "baselines use the seed-42 negatives. Smallest attainable p = "
          f"{fmt_p(2 / (n + 1))}.*", "",
          "| Block | Protocol | Comparison | Delta MRR | 95% CI | p | p (Holm) |",
          "|---|---|---|---|---|---|---|"]
    tex = [r"\begin{tabular}{lllrlll}", r"\hline",
           r"Block & Protocol & Comparison & $\Delta$MRR & 95\% CI & $p$ & $p_{\mathrm{Holm}}$ \\",
           r"\hline"]
    for r in rows:
        ci = f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]"
        md.append(f"| {r['block']} | {r['protocol']} | {r['comparison']} | "
                  f"{r['delta']:+.4f} | {ci} | {fmt_p(r['p'])} | {fmt_p(r['p_holm'])} |")
        tex.append(f"{r['block']} & {r['protocol']} & {r['comparison']} & "
                   f"{r['delta']:+.4f} & {ci} & {fmt_p(r['p'])} & {fmt_p(r['p_holm'])} \\\\")
    tex += [r"\hline", r"\end{tabular}"]
    (TABLES / "tableS6_pvalues.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (TABLES / "tableS6_pvalues.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")

    log(f"done in {time.time() - t0:.0f}s; "
        f"{sum(r['significant'] for r in rows)}/{len(rows)} significant after Holm")
    for r in rows:
        if not r["significant"]:
            log(f"  not significant: [{r['protocol']}] {r['comparison']}  "
                f"delta={r['delta']:+.4f}  p={fmt_p(r['p'])}  holm={fmt_p(r['p_holm'])}")
    for x in retained:
        log(f"  retained share [{x['protocol']}] {x['method']:22s} "
            f"{x['retained_share']:.3f} [{x['ci_low']:.3f}, {x['ci_high']:.3f}]")


if __name__ == "__main__":
    main()
