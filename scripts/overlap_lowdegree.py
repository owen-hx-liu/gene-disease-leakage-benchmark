"""Direct train--test overlap among low-degree diseases (S8 Table).

A held-out pair has direct overlap when the R0 training graph joins its gene and disease
by an edge of any relation, in either direction (the edges R1 removes). This script splits
the full-ranking R0 MRR of the lowest-degree test edges into overlap and non-overlap pairs,
and reports R1 and R2 beside them. It reads only frozen result files (no training, no
rescoring) and reuses the loaders and quartile rule of make_revision_assets.py.

Writes revision/tables/tableS8_overlap_lowdegree.tex and
revision/overlap_lowdegree_numbers.json.

Usage:  python scripts/overlap_lowdegree.py
"""
from __future__ import annotations

import csv
import json

import numpy as np

import make_revision_assets as mra
from make_revision_assets import KGE, LABEL, SPLITS, OUT, fmt, mrr, write

STRUCT = ["CommonNeighbors", "AdamicAdar", "Jaccard"] + KGE


def overlap_mask(genes, dis) -> np.ndarray:
    test = set(zip(genes, dis))
    hit = set()
    with (SPLITS / "train.csv").open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            a, b = row["source_id"], row["target_id"]
            if (a, b) in test:
                hit.add((a, b))
            elif (b, a) in test:
                hit.add((b, a))
    return np.array([(g, d) in hit for g, d in zip(genes, dis)], bool)


def main():
    rr = mra.per_edge()
    genes, dis = mra.test_edges()
    deg = mra.degrees()
    ddeg = np.array([deg.get(d, 0) for d in dis])
    ov = overlap_mask(genes, dis)
    # disease quartiles exactly as in Table 5
    q1 = np.sort(np.array_split(np.argsort(ddeg, kind="stable"), 4)[0])
    b15 = np.where((ddeg >= 1) & (ddeg <= 5))[0]

    N = {"n_test": len(dis), "n_overlap": int(ov.sum())}
    rows = []
    for m in STRUCT:
        c = [LABEL[m]]
        for name, b in (("degree1-5", b15), ("Q1", q1)):
            a0 = rr[("full", m, "R0")]
            bo, bn = b[ov[b]], b[~ov[b]]
            share = float(a0[:, bo].sum(axis=1).mean() / a0[:, b].sum(axis=1).mean())
            v = {"n": int(b.size), "n_overlap": int(bo.size),
                 "R0": mrr(a0, b), "R0_overlap": mrr(a0, bo), "R0_no_overlap": mrr(a0, bn),
                 "R0_share_from_overlap": share,
                 "R1": mrr(rr[("full", m, "R1")], b), "R2": mrr(rr[("full", m, "R2")], b)}
            N.setdefault(m, {})[name] = v
            c += [fmt(v["R0"]), fmt(v["R0_overlap"]), fmt(v["R0_no_overlap"]),
                  f"{100 * share:.0f}\\%", fmt(v["R1"]), fmt(v["R2"])]
        rows.append(" & ".join(c) + r" \\")
        if m == "Jaccard":
            rows.append(r"\midrule")
    n15, no15 = b15.size, int(ov[b15].sum())
    nq, noq = q1.size, int(ov[q1].sum())
    write("tableS8_overlap_lowdegree.tex", r"""\begin{tabular}{l cccccc cccccc}
\toprule
 & \multicolumn{6}{c}{Disease degree 1--5 ($n$ = """ + f"{n15:,}; overlap {no15:,}" + r""")} & \multicolumn{6}{c}{Disease Q1 ($n$ = """ + f"{nq:,}; overlap {noq:,}" + r""")} \\
\cmidrule(lr){2-7}\cmidrule(lr){8-13}
 & \multicolumn{4}{c}{R0} & & & \multicolumn{4}{c}{R0} & & \\
\cmidrule(lr){2-5}\cmidrule(lr){8-11}
Method & All & Overlap & No overlap & Share & R1 & R2 & All & Overlap & No overlap & Share & R1 & R2 \\
\midrule
""" + "\n".join(rows) + "\n" + r"""\bottomrule
\end{tabular}""")
    (OUT / "overlap_lowdegree_numbers.json").write_text(json.dumps(N, indent=1), encoding="utf-8")
    print(json.dumps({m: {k: {kk: round(vv, 3) if isinstance(vv, float) else vv
                              for kk, vv in v.items()} for k, v in N[m].items()} for m in STRUCT},
                     indent=1))


if __name__ == "__main__":
    main()
