"""Justify the 2,000 neighbor-degree cap used by Common Neighbors and Adamic-Adar (S1 Text).

Two measurements, both on the VALIDATION edges so the test edges never inform the cap:

  1. Enrichment by neighbor degree. For each true validation gene-disease pair and for an
     equal number of random gene-disease pairs (same genes, diseases drawn uniformly from
     the disease pool, known pairs excluded), count the shared neighbors in each band of
     neighbor degree. Enrichment = mean count per true pair / mean count per random pair.
     A band with enrichment near 1 carries no information about whether a pair is true.
  2. Cap sweep. Rank every validation edge against 50 type-matched negatives (the same
     harness and seed as the main runs) with CN and AA at several caps, and report MRR.

Reads the R0 training graph and splits only; writes
  data/processed/results/hub_cap/hub_cap_analysis.json

Usage:  python scripts/hub_cap_analysis.py
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib_eval import (_category_pools, _load_edges, load_regime,  # noqa: E402
                      rank_test_edges)
from run_baselines import build_adjacency  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "data" / "processed" / "splits"
OUT = ROOT / "data" / "processed" / "results" / "hub_cap" / "hub_cap_analysis.json"

BANDS = [(2, 10), (11, 100), (101, 500), (501, 1000), (1001, 2000), (2001, 5000),
         (5001, 10**9)]
CAPS = [250, 500, 1000, 1500, 2000, 3000, 5000, 10**9]
SEED = 42
RANDOM_PER_TRUE = 20  # random pairs drawn per true pair, so rare bands get a finite estimate


def band_of(d: int) -> int:
    for i, (lo, hi) in enumerate(BANDS):
        if lo <= d <= hi:
            return i
    return -1  # degree 1 cannot be shared by two distinct nodes


def shared_band_counts(adj, deg, g, d) -> np.ndarray:
    out = np.zeros(len(BANDS))
    ng, nd = adj.get(g), adj.get(d)
    if not ng or not nd:
        return out
    if len(ng) > len(nd):
        ng, nd = nd, ng
    for w in ng:
        if w in nd:
            b = band_of(deg[w])
            if b >= 0:
                out[b] += 1
    return out


def main():
    t0 = time.time()
    reg = load_regime("R0")
    train, test = reg.train_edges, reg.test_edges
    valid = _load_edges(SPLITS / "valid.csv")
    print(f"train {len(train):,}  valid {len(valid):,}  test {len(test):,}", flush=True)
    adj, deg = build_adjacency(train)
    print(f"adjacency built in {time.time() - t0:.0f}s", flush=True)

    pools = _category_pools(train, reg.hub_set if reg.hub_filter else frozenset())
    dis_pool = pools["disease"]
    # every disease known for a validation gene anywhere (train, valid or test) is filtered
    genes = set(valid[:, 0])
    known: dict = {}
    for arr in (train, valid, test):
        for h, t in zip(arr[:, 0].tolist(), arr[:, 2].tolist()):
            if h in genes:
                known.setdefault(h, set()).add(t)
    print(f"filter sets built at {time.time() - t0:.0f}s", flush=True)

    # 1. enrichment
    rng = np.random.default_rng(SEED)
    true_c = np.zeros(len(BANDS))
    rand_c = np.zeros(len(BANDS))
    true_any = rand_any = 0
    for g, _r, d in valid:
        c = shared_band_counts(adj, deg, g, d)
        true_c += c
        true_any += c.sum() > 0
        for _ in range(RANDOM_PER_TRUE):
            while True:
                dr = dis_pool[rng.integers(len(dis_pool))]
                if dr not in known.get(g, ()):
                    break
            c = shared_band_counts(adj, deg, g, dr)
            rand_c += c
            rand_any += c.sum() > 0
    n = len(valid)
    n_rand = n * RANDOM_PER_TRUE
    enrich = []
    for (lo, hi), tc, rc in zip(BANDS, true_c, rand_c):
        enrich.append({"band": f"{lo}-{hi}" if hi < 10**9 else f">{lo - 1}",
                       "shared_per_true_pair": tc / n, "shared_per_random_pair": rc / n_rand,
                       "random_pairs_sharing_count": int(rc),
                       "enrichment": (tc / n) / (rc / n_rand) if rc > 0 else None})
        print(f"  band {enrich[-1]['band']:>10}: true {tc / n:8.3f}  random {rc / n_rand:8.4f}  "
              f"enrichment {enrich[-1]['enrichment']}", flush=True)
    print(f"enrichment done at {time.time() - t0:.0f}s", flush=True)

    # 2. cap sweep on validation edges, sampled ranking, seed-42 negatives
    log_ = math.log
    sweep = []
    for cap in CAPS:
        def cn(g, d, cap=cap):
            ng, nd = adj.get(g), adj.get(d)
            if not ng or not nd:
                return 0.0
            if len(ng) > len(nd):
                ng, nd = nd, ng
            return float(sum(1 for w in ng if w in nd and deg[w] <= cap))

        def aa(g, d, cap=cap):
            ng, nd = adj.get(g), adj.get(d)
            if not ng or not nd:
                return 0.0
            if len(ng) > len(nd):
                ng, nd = nd, ng
            s = 0.0
            for w in ng:
                if w in nd:
                    dw = deg[w]
                    if 1 < dw <= cap:
                        s += 1.0 / log_(dw)
            return s

        row = {"cap": cap if cap < 10**9 else None}
        for name, fn in (("CommonNeighbors", cn), ("AdamicAdar", aa)):
            ranks = rank_test_edges(fn, train, valid, reg.hub_set, reg.hub_filter, n_neg=50,
                                    seed=SEED, pools=pools, known=known)
            row[name] = float(np.mean(1.0 / ranks))
        sweep.append(row)
        print(f"  cap {row['cap']}: CN {row['CommonNeighbors']:.4f}  AA {row['AdamicAdar']:.4f}"
              f"  ({time.time() - t0:.0f}s)", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "description": "Neighbor-degree cap justification for CN/AA (S1 Text): shared-neighbor "
                       "enrichment by neighbor degree, and a cap sweep, on validation edges.",
        "graph": "R0 training graph", "edges_scored": "validation (valid.csv)",
        "n_pairs": n, "seed": SEED, "protocol": "sampled, 50 type-matched negatives",
        "true_pairs_with_any_shared_neighbor": true_any / n,
        "random_pairs_with_any_shared_neighbor": rand_any / n_rand,
        "random_pairs_per_true_pair": RANDOM_PER_TRUE,
        "enrichment_by_neighbor_degree": enrich, "cap_sweep": sweep,
    }, indent=1), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
