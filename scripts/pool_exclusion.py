"""R1 and R3 with and without test edges whose true disease leaves the pool (S9 Table).

The full-ranking candidate pool of a regime is every disease with at least one edge in
that regime's training graph, so R1 and R3 drop the diseases they leave with no edge. A
test edge whose true disease is one of them is kept: the KGE models rank it last and the
baselines score it zero. This script reports each method's R1 and R3 change from R0 on
all test edges and with those edges excluded, with a paired bootstrap on the excluded set.
It reads only frozen result files and the split CSVs (no training, no rescoring).

Writes revision/tables/tableS9_pool_exclusion.tex and revision/pool_exclusion_numbers.json.

Usage:  python scripts/pool_exclusion.py
"""
from __future__ import annotations

import csv
import json

import numpy as np

import make_revision_assets as mra
from make_revision_assets import KGE, LABEL, SPLITS, OUT, pfmt, signed, write
from lib_eval import paired_bootstrap_pvalue

METHODS = ["CommonNeighbors", "AdamicAdar", "Jaccard", "PreferentialAttachment"] + KGE
TRAIN = {"R1": "train_R1_redundancy.csv", "R3": "train_R3_orthology_blocked.csv"}
N_BOOT = 100_000


def nodes_in(fname) -> set[str]:
    out = set()
    with (SPLITS / fname).open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out.add(row["source_id"])
            out.add(row["target_id"])
    return out


def pct(x):
    return f"{100 * x:+.1f}\\%".replace("-", "$-$")


def main():
    rr = mra.per_edge()
    _, dis = mra.test_edges()
    absent = {}
    for r, f in TRAIN.items():
        present = nodes_in(f)
        absent[r] = np.array([d not in present for d in dis], bool)
    N = {r: {"n_excluded": int(a.sum())} for r, a in absent.items()}

    panels = {}
    for proto in ("full", "sampled"):
        rows = []
        for m in METHODS:
            c = [LABEL[m]]
            for r in ("R1", "R3"):
                a0, a1 = rr[(proto, m, "R0")], rr[(proto, m, r)]
                keep = ~absent[r]
                ch_all = mra.mrr(a1) / mra.mrr(a0) - 1
                ch_keep = mra.mrr(a1, keep) / mra.mrr(a0, keep) - 1
                # paired tests as in the main family: KGE ranks averaged over seeds,
                # sampled-ranking baselines on the seed-42 negatives (row 0)
                if m in KGE or proto == "full":
                    x0, x1 = a0.mean(axis=0), a1.mean(axis=0)
                else:
                    x0, x1 = a0[0], a1[0]
                t = paired_bootstrap_pvalue(x1[keep], x0[keep], n=N_BOOT, seed=42)
                N[r][f"{proto}|{m}"] = {"change_all": ch_all, "change_excluded": ch_keep,
                                        "delta": t["delta"], "ci_low": t["ci_low"],
                                        "ci_high": t["ci_high"], "p": t["p_value_plus_one"]}
                c += [pct(ch_all), pct(ch_keep),
                      f"{signed(t['delta'])} ({signed(t['ci_low'])} to {signed(t['ci_high'])})",
                      pfmt(t["p_value_plus_one"])]
            rows.append(" & ".join(c) + r" \\")
            if m == "PreferentialAttachment":
                rows.append(r"\midrule")
        panels[proto] = "\n".join(rows)

    head = (r" & \multicolumn{4}{c}{R1 ($" + f"{N['R1']['n_excluded']}" + r"$ edges excluded)}"
            r" & \multicolumn{4}{c}{R3 ($" + f"{N['R3']['n_excluded']}" + r"$ edges excluded)} \\"
            "\n" r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}" "\n"
            r"Method & All & Excl. & Difference, excl.\ (95\% CI) & $p$"
            r" & All & Excl. & Difference, excl.\ (95\% CI) & $p$ \\")
    write("tableS9_pool_exclusion.tex", r"""\begin{tabular}{l cccc cccc}
\toprule
\multicolumn{9}{l}{\textit{Panel A. Full ranking}} \\
\midrule
""" + head + "\n" + r"\midrule" + "\n" + panels["full"] + "\n" + r"""\midrule
\multicolumn{9}{l}{\textit{Panel B. Sampled ranking}} \\
\midrule
""" + head + "\n" + r"\midrule" + "\n" + panels["sampled"] + "\n" + r"""\bottomrule
\end{tabular}""")
    (OUT / "pool_exclusion_numbers.json").write_text(json.dumps(N, indent=1), encoding="utf-8")
    print(json.dumps(N, indent=1))


if __name__ == "__main__":
    main()
