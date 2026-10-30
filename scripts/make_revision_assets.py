"""Build the revision's tables and figures from the frozen result JSONs.

Reads only files already on disk (no training, no rescoring), so every number in the
revised manuscript traces to a result file. Writes to revision/tables and
revision/figures, and dumps every computed quantity to revision/revision_numbers.json
so text claims can be checked against one file.

Inputs
  results/baselines/baselines_R*.json        sampled ranking, baselines, per-edge RR x 3 seeds
  results/kge/kge_<m>_<R>_seed*.json          sampled ranking, KGE, per-edge RR
  results/robustness/robustness_R*.json       full ranking, baselines, per-edge RR
  results/kge_fullrank/kge_<m>_<R>_seed*.json full ranking, KGE, per-edge RR
  results/stats/multiple_testing.json         Holm-adjusted paired bootstrap tests
  results/null/degree_null.json               200-graph permutation null (baselines, sampled)
  results/hetionet/baselines_R*.json          Hetionet replication (sampled)
  splits/test.csv, splits/node_degree.csv,
  splits/train_R2_degree_null_seed42.csv      per-edge degrees, recreated test pairs

Usage:  python scripts/make_revision_assets.py
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "data" / "processed" / "results"
SPLITS = ROOT / "data" / "processed" / "splits"
OUT = ROOT / "revision"
TAB = OUT / "tables"
FIG = OUT / "figures"

BASELINES = ["Random", "CommonNeighbors", "AdamicAdar", "Jaccard", "PreferentialAttachment"]
KGE = ["TransE", "RotatE", "DistMult", "ComplEx"]
METHODS = BASELINES + KGE
STRUCT = ["CommonNeighbors", "AdamicAdar", "Jaccard"] + KGE
REGIMES = ["R0", "R1", "R2", "R3"]
SEEDS = [42, 1, 7]
POOL = {"R0": 14307, "R1": 14252, "R2": 14307, "R3": 14137}

LABEL = {"Random": "Random", "CommonNeighbors": "Common Neighbors", "AdamicAdar": "Adamic--Adar",
         "Jaccard": "Jaccard", "PreferentialAttachment": "Pref. Attachment",
         "TransE": "TransE", "RotatE": "RotatE", "DistMult": "DistMult", "ComplEx": "ComplEx"}
PLAIN = {k: v.replace("--", "-") for k, v in LABEL.items()}
SHORT = {"Random": "Random", "CommonNeighbors": "CN", "AdamicAdar": "AA", "Jaccard": "Jaccard",
         "PreferentialAttachment": "PA", "TransE": "TransE", "RotatE": "RotatE",
         "DistMult": "DistMult", "ComplEx": "ComplEx"}
# names used inside multiple_testing.json
MT_NAME = {"CommonNeighbors": "Common Neighbors", "AdamicAdar": "Adamic-Adar", "Jaccard": "Jaccard",
           "PreferentialAttachment": "PA", "TransE": "TransE", "RotatE": "RotatE",
           "DistMult": "DistMult", "ComplEx": "ComplEx"}

# same Okabe-Ito map as scripts/make_figures.py, so a method keeps its colour
COLOR = {"Random": "#999999", "CommonNeighbors": "#56B4E9", "AdamicAdar": "#0072B2",
         "Jaccard": "#009E73", "PreferentialAttachment": "#E69F00", "TransE": "#D55E00",
         "RotatE": "#CC79A7", "DistMult": "#000000", "ComplEx": "#785EF0"}
MARKER = {"Random": "X", "CommonNeighbors": "o", "AdamicAdar": "s", "Jaccard": "^",
          "PreferentialAttachment": "D", "TransE": "P", "RotatE": "v", "DistMult": "*",
          "ComplEx": "h"}

plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "font.size": 8.5, "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#DDDDDD", "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False,
})


def load(p: Path) -> dict:
    if not p.exists():
        sys.exit(f"[make_revision_assets] missing input: {p.relative_to(ROOT)}")
    return json.loads(p.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- loading
def per_edge() -> dict:
    """{(protocol, method, regime): (n_seeds_or_1, 4228) array of reciprocal ranks}."""
    rr = {}
    for r in REGIMES:
        b = load(RES / "baselines" / f"baselines_{r}.json")
        for m, lists in b["per_edge_reciprocal_ranks"].items():
            rr[("sampled", m, r)] = np.asarray(lists, float)
        f = load(RES / "robustness" / f"robustness_{r}.json")
        for m, lst in f["per_edge_reciprocal_ranks"].items():
            rr[("full", m, r)] = np.asarray(lst, float)[None, :]
        for m in KGE:
            rr[("sampled", m, r)] = np.vstack([np.asarray(load(
                RES / "kge" / f"kge_{m}_{r}_seed{s}.json")["reciprocal_ranks"], float) for s in SEEDS])
            rr[("full", m, r)] = np.vstack([np.asarray(load(
                RES / "kge_fullrank" / f"kge_{m}_{r}_seed{s}.json")["reciprocal_ranks"], float)
                for s in SEEDS])
    return rr


def full_random_mrr(n: int) -> float:
    return sum(1.0 / k for k in range(1, n + 1)) / n


def test_edges() -> tuple[list[str], list[str]]:
    g, d = [], []
    with (SPLITS / "test.csv").open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            g.append(row["source_id"])
            d.append(row["target_id"])
    return g, d


def degrees() -> dict[str, int]:
    deg = {}
    with (SPLITS / "node_degree.csv").open(encoding="utf-8") as fh:
        rd = csv.reader(fh)
        next(rd)
        for row in rd:
            if len(row) >= 2 and row[0]:
                deg[row[0]] = int(row[1])
    return deg


def recreated_mask(genes, dis) -> np.ndarray:
    test = set(zip(genes, dis))
    hit = set()
    with (SPLITS / "train_R2_degree_null_seed42.csv").open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["relation"] == "BIOLINK:GENE_ASSOCIATED_WITH_CONDITION":
                p = (row["source_id"], row["target_id"])
                if p in test:
                    hit.add(p)
    return np.array([(a, b) in hit for a, b in zip(genes, dis)], bool)


# --------------------------------------------------------------------------- helpers
def mrr(a: np.ndarray, idx=None) -> float:
    """Seed-mean MRR (rows = seeds)."""
    x = a if idx is None else a[:, idx]
    return float(x.mean(axis=1).mean())


def mrr_sd(a: np.ndarray) -> float:
    return float(a.mean(axis=1).std(ddof=1)) if a.shape[0] > 1 else 0.0


def hits(a, k):
    return float((1.0 / a <= k).mean(axis=1).mean())


def median_rank(a):
    return float(np.mean([np.median(1.0 / row) for row in a]))


def fmt(x, d=3):
    return f"{x:.{d}f}"


def pct(x, d=0):
    return f"{100 * x:.{d}f}\\%"


def signed(x, d=3):
    s = f"{x:+.{d}f}"
    return s.replace("-", "$-$")


def star(p):
    if p is None:
        return ""
    if p < 0.01:
        return "$^{**}$"
    if p < 0.05:
        return "$^{*}$"
    return ""


def pfmt(p):
    if p >= 0.995:
        return "1.00"
    if p < 0.002:
        return "$<$0.002"
    if p < 0.01:
        return f"{p:.3f}"
    return f"{p:.2f}"


def write(name: str, text: str):
    (TAB / name).write_text(text, encoding="utf-8")
    print(f"  wrote tables/{name}")


# PLOS ONE figure files: TIFF, RGB, LZW, 300-600 dpi, at most 7.5 x 8.75 in. We match the
# first submission's files: 600 dpi and at most 4,440 px (7.4 in) wide.
PLOS_DPI = 600
PLOS_MAX_W, PLOS_MAX_H = 4440, int(8.75 * PLOS_DPI)
PLOS_NAME = {"fig1_regime_schematic": "Fig1", "fig2_metrics_protocols": "Fig2",
             "fig3_regimes": "Fig3", "fig4_retained_vs_pa": "Fig4",
             "fig5_degree_quartiles": "Fig5"}


def savefig(fig, stem):
    from io import BytesIO
    from PIL import Image
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{stem}.{ext}", bbox_inches="tight")
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=PLOS_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    im = Image.open(buf).convert("RGB")
    scale = min(1.0, PLOS_MAX_W / im.width, PLOS_MAX_H / im.height)
    if scale < 1.0:
        im = im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)
    (FIG / "plos").mkdir(exist_ok=True)
    name = PLOS_NAME.get(stem, stem)
    im.save(FIG / "plos" / f"{name}.tif", compression="tiff_lzw", dpi=(PLOS_DPI, PLOS_DPI))
    print(f"  wrote figures/{stem}.pdf/.png and figures/plos/{name}.tif "
          f"({im.width}x{im.height} px = {im.width / PLOS_DPI:.2f} x {im.height / PLOS_DPI:.2f} in)")


# --------------------------------------------------------------------------- main compute
def main():
    TAB.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    rr = per_edge()
    mt = load(RES / "stats" / "multiple_testing.json")
    tests = {(t["protocol"], t["comparison"]): t for t in mt["tests"]}
    rshare = {(r["protocol"], r["method"]): r for r in mt["retained_share"]}
    null = load(RES / "null" / "degree_null.json")["methods"]
    kge_sum = load(RES / "kge" / "kge_summary.json")["aggregate"]
    bl_s = {r: load(RES / "baselines" / f"baselines_{r}.json") for r in REGIMES}
    genes, dis = test_edges()
    deg = degrees()
    gdeg = np.array([deg.get(g, 0) for g in genes])
    ddeg = np.array([deg.get(d, 0) for d in dis])
    rec = recreated_mask(genes, dis)
    N = {}  # every number, dumped at the end

    def regime_p(proto, m, r):
        t = tests.get((proto, f"{MT_NAME[m]}: {r} vs R0"))
        return None if t is None else t["p_holm"]

    def pa_test(proto, m):
        return tests.get((proto, f"{MT_NAME[m]} vs PA at R2"))

    # ---- aggregate MRR per protocol/method/regime
    agg = {}
    for proto in ("sampled", "full"):
        for m in METHODS:
            for r in REGIMES:
                if m == "Random" and proto == "full":
                    v, sd = full_random_mrr(POOL[r]), 0.0
                elif m in BASELINES and proto == "sampled":
                    v = bl_s[r]["methods"][m]["MRR"]["mean"]
                    sd = bl_s[r]["methods"][m]["MRR"]["sd"]
                else:
                    a = rr[(proto, m, r)]
                    v, sd = mrr(a), mrr_sd(a)
                agg[(proto, m, r)] = (v, sd)
    N["mrr"] = {f"{p}|{m}|{r}": v for (p, m, r), v in agg.items()}

    # ---------------------------------------------------------------- Table 3
    rows_a, rows_b = [], []
    for m in METHODS:
        cells = [LABEL[m]]
        for r in REGIMES:
            v, sd = agg[("full", m, r)]
            s = fmt(v) if (m in BASELINES) else f"{fmt(v)} $\\pm$ {fmt(sd)}"
            if r != "R0" and m != "Random":
                s += star(regime_p("full", m, r))
            cells.append(s)
        rows_a.append(" & ".join(cells) + r" \\")
        cells = [LABEL[m]]
        if m == "Random":
            rnd = load(RES / "robustness" / "robustness_R0.json")["methods"]["Random"]
            cells += [fmt(rnd["Hits@1"]), fmt(rnd["Hits@10"]), fmt(rnd["Hits@100"]),
                      f"{rnd['MedianRank']:,.0f}"]
        else:
            a = rr[("full", m, "R0")]
            cells += [fmt(hits(a, 1)), fmt(hits(a, 10)), fmt(hits(a, 100)), f"{median_rank(a):,.0f}"]
        v, _ = agg[("sampled", m, "R0")]
        if m in BASELINES:
            h10 = bl_s["R0"]["methods"][m]["Hits@10"]["mean"]
            au = bl_s["R0"]["methods"][m]["AUROC"]["mean"]
        else:
            h10 = kge_sum[f"{m}|R0|d64|e300"]["Hits@10_mean"]
            au = kge_sum[f"{m}|R0|d64|e300"]["AUROC_type_mean"]
        cells += [fmt(v), fmt(h10), fmt(au)]
        rows_b.append(" & ".join(cells) + r" \\")
        if m == "PreferentialAttachment":
            rows_a.append(r"\midrule")
            rows_b.append(r"\midrule")
    write("table3_regimes_fullrank.tex", r"""\begin{tabular}{l llll}
\multicolumn{5}{l}{\textit{Panel A. Full-ranking MRR in every regime}} \\
\toprule
Method & R0 & R1 & R2$^{a}$ & R3 \\
\midrule
""" + "\n".join(rows_a) + "\n" + r"""\bottomrule
\end{tabular}

\vspace{1.4em}

\begin{tabular}{l cccr ccc}
\multicolumn{8}{l}{\textit{Panel B. R0 performance under both protocols}} \\
\toprule
 & \multicolumn{4}{c}{Full ranking} & \multicolumn{3}{c}{Sampled ranking} \\
\cmidrule(lr){2-5}\cmidrule(lr){6-8}
Method & Hits@1 & Hits@10 & Hits@100 & Median rank & MRR & Hits@10 & AUROC \\
\midrule
""" + "\n".join(rows_b) + "\n" + r"""\bottomrule
\end{tabular}""")

    # ---------------------------------------------------------------- Table 4
    panels = {}
    for proto in ("full", "sampled"):
        rows = []
        for m in METHODS[1:]:
            r0, r2 = agg[(proto, m, "R0")][0], agg[(proto, m, "R2")][0]
            rs = rshare[(proto, m)]
            c = [LABEL[m], fmt(r0), fmt(r2),
                 f"{100*rs['retained_share']:.0f}\\% ({100*rs['ci_low']:.0f}--{100*rs['ci_high']:.0f})"]
            if m == "PreferentialAttachment":
                c.append("--")
            else:
                t = pa_test(proto, m)
                c.append(f"{signed(t['delta'])} ({pfmt(t['p_holm'])})")
            if proto == "sampled":
                if m in BASELINES:
                    n = null[m]
                    z = n["real_minus_null_mean"] / n["null_MRR_sd"]
                    c += [f"{n['p_value_plus_one']:.3f}", f"{z:+.0f}".replace("-", "$-$")]
                else:
                    c += ["n/a", "n/a"]
            rows.append(" & ".join(c) + r" \\")
            if m == "PreferentialAttachment":
                rows.append(r"\midrule")
        panels[proto] = "\n".join(rows)
    head = (r"Method & R0 & R2 & Retained (95\% CI) & "
            r"$m_X(\mathrm{R2}) - m_{\mathrm{PA}}(\mathrm{R2})$ ($p_{\mathrm{Holm}}$)")
    write("table4_rewiring.tex", r"""\begin{tabular}{l ccc c}
\multicolumn{5}{l}{\textit{Panel A. Full ranking}} \\
\toprule
""" + head + r""" \\
\midrule
""" + panels["full"] + "\n" + r"""\bottomrule
\end{tabular}

\vspace{1.4em}

\begin{tabular}{l ccc c cc}
\multicolumn{7}{l}{\textit{Panel B. Sampled ranking}} \\
\toprule
 & & & & & \multicolumn{2}{c}{200 rewired graphs$^{b}$} \\
\cmidrule(lr){6-7}
""" + head + r""" & Perm.\ $p$ & $z$ \\
\midrule
""" + panels["sampled"] + "\n" + r"""\bottomrule
\end{tabular}""")
    N["retained_share"] = {f"{k[0]}|{k[1]}": v for k, v in rshare.items()}

    # ---------------------------------------------------------------- degree stratification
    def strata(x):
        order = np.argsort(x, kind="stable")
        ch = np.array_split(order, 4)
        return ch, [(int(x[c].min()), int(x[c].max())) for c in ch]

    dq, drng = strata(ddeg)
    gq, grng = strata(gdeg)
    N["disease_quartile_ranges"] = drng
    N["gene_quartile_ranges"] = grng
    strat = {}
    for proto in ("full", "sampled"):
        for m in METHODS:
            if m == "Random" and proto == "full":
                continue
            for r in ("R0", "R2"):
                a = rr[(proto, m, r)]
                tot = a.sum(axis=1).mean()
                strat[(proto, m, r)] = {
                    "dQ": [mrr(a, c) for c in dq], "gQ": [mrr(a, c) for c in gq],
                    "dShare": [float(a[:, c].sum(axis=1).mean() / tot) for c in dq]}
    N["strat"] = {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in strat.items()}

    def strat_table(proto):
        rows = []
        ms = METHODS[1:] if proto == "full" else METHODS
        for m in ms:
            s0, s2 = strat[(proto, m, "R0")], strat[(proto, m, "R2")]
            d1 = s2["dQ"][0] / s0["dQ"][0] - 1 if s0["dQ"][0] > 0 else float("nan")
            d4 = s2["dQ"][3] / s0["dQ"][3] - 1 if s0["dQ"][3] > 0 else float("nan")

            def p(v):
                return "n/a" if math.isnan(v) else f"{100*v:+.0f}\\%".replace("-", "$-$")
            rows.append(" & ".join([
                LABEL[m], fmt(s0["dQ"][0]), fmt(s0["dQ"][3]), fmt(s0["gQ"][0]), fmt(s0["gQ"][3]),
                p(d1), p(d4), f"{100*s2['dShare'][0]:.0f}\\%", f"{100*s2['dShare'][3]:.0f}\\%"]) + r" \\")
            if m == "PreferentialAttachment":
                rows.append(r"\midrule")
        return r"""\begin{tabular}{l cc cc cc cc}
\toprule
 & \multicolumn{2}{c}{R0 MRR, disease degree} & \multicolumn{2}{c}{R0 MRR, gene degree} & \multicolumn{2}{c}{R2 change in MRR} & \multicolumn{2}{c}{Share of R2 score} \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}
Method & Q1 & Q4 & Q1 & Q4 & Disease Q1 & Disease Q4 & Disease Q1 & Disease Q4 \\
\midrule
""" + "\n".join(rows) + "\n" + r"""\bottomrule
\end{tabular}"""
    write("table5_degree_fullrank.tex", strat_table("full"))
    write("tableS7_degree_sampled.tex", strat_table("sampled"))

    # ---------------------------------------------------------------- Table 6
    def bucket(lo, hi):
        return np.where((ddeg >= lo) & (ddeg <= hi))[0]
    b15, b301 = bucket(1, 5), bucket(301, 10**9)
    N["bucket_n"] = {"1-5": int(b15.size), "301+": int(b301.size)}
    rowsA = []
    for m in KGE + ["PreferentialAttachment"]:
        c = [LABEL[m]]
        for b in (b15, b301):
            v0, v2 = mrr(rr[("full", m, "R0")], b), mrr(rr[("full", m, "R2")], b)
            c += [fmt(v0), fmt(v2), (f"{100*(v2/v0-1):+.0f}\\%".replace("-", "$-$") if v0 > 0 else "n/a")]
        rowsA.append(" & ".join(c) + r" \\")
        N.setdefault("table6A", {})[m] = c[1:]
    ex = [("HGNC:21553", "MONDO:0013600", "\\gene{LIN28B-AS1} $\\to$ insomnia"),
          ("HGNC:10457", "MONDO:0004579", "\\gene{RS1} $\\to$ retinoschisis"),
          ("HGNC:7765", "MONDO:0003304", "\\gene{NF1} $\\to$ plexiform neurofibroma"),
          ("HGNC:6342", "MONDO:0000544", "\\gene{KIT} $\\to$ mucosal melanoma"),
          ("HGNC:8138", "MONDO:0016391", "\\gene{ONECUT1} $\\to$ neonatal diabetes")]
    cols = KGE + ["PreferentialAttachment", "AdamicAdar"]
    rowsB = []
    for g, d, lab in ex:
        i = [k for k in range(len(genes)) if genes[k] == g and dis[k] == d]
        if len(i) != 1:
            sys.exit(f"worked example {g}->{d} not found uniquely")
        i = i[0]
        rk = [float(np.mean(1.0 / rr[("full", m, "R0")][:, i])) for m in cols]
        N.setdefault("table6B", {})[lab] = rk
        rowsB.append(" & ".join([lab, f"{gdeg[i]:,}", f"{ddeg[i]:,}"] + [f"{x:,.0f}" for x in rk]) + r" \\")
    ins = [k for k in range(len(dis)) if dis[k] == "MONDO:0013600"]
    med = [float(np.median(np.mean(1.0 / rr[("full", m, "R0")][:, ins], axis=0))) for m in cols]
    N["table6B"]["insomnia_median"] = med
    N["insomnia_n"] = len(ins)
    rowsB.append(" & ".join([f"Insomnia, median over its {len(ins)} test genes", "--", "801"] +
                            [f"{x:,.0f}" for x in med]) + r" \\")
    write("table6_examples_fullrank.tex", r"""\begin{tabular}{l ccc ccc}
\toprule
\multicolumn{7}{l}{\textit{Panel A. Full-ranking MRR, R0 $\to$ R2, by training degree of the true disease}} \\
\midrule
 & \multicolumn{3}{c}{Disease degree 1--5 ($n$ = """ + f"{b15.size}" + r""")} & \multicolumn{3}{c}{Disease degree $\geq$301 ($n$ = """ + f"{b301.size:,}" + r""")} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}
Method & R0 & R2 & Change & R0 & R2 & Change \\
\midrule
""" + "\n".join(rowsA) + "\n" + r"""\bottomrule
\end{tabular}

\vspace{1.2em}

\begin{tabular}{l rr rrrr rr}
\toprule
\multicolumn{9}{l}{\textit{Panel B. Rank of the true disease among 14,307 candidates at R0 (1 = top)}} \\
\midrule
Held-out edge (gene $\to$ disease) & Gene deg. & Disease deg. & TransE & RotatE & DistMult & ComplEx & PA & AA \\
\midrule
""" + "\n".join(rowsB) + "\n" + r"""\bottomrule
\end{tabular}""")

    # ---------------------------------------------------------------- Table 7 (Hetionet)
    het = {r: load(RES / "hetionet" / f"baselines_{r}.json") for r in ("R0", "R1", "R2")}
    rows = []
    for m in BASELINES:
        v = {r: het[r]["methods"][m]["MRR"] for r in het}
        rows.append(" & ".join([LABEL[m]] + [f"{fmt(v[r]['mean'])} $\\pm$ {fmt(v[r]['sd'])}" for r in ("R0", "R1", "R2")]
                               + [f"{100*v['R2']['mean']/v['R0']['mean']:.1f}\\%"]) + r" \\")
        N.setdefault("hetionet", {})[m] = {r: v[r]["mean"] for r in v}
    write("table7_hetionet.tex", r"""\begin{tabular}{lcccc}
\toprule
Method & MRR R0 & MRR R1 & MRR R2 & Retained (R2/R0) \\
\midrule
""" + "\n".join(rows) + "\n" + r"""\bottomrule
\end{tabular}""")

    # ---------------------------------------------------------------- S1 Table (sampled grid)
    rows = []
    for m in METHODS:
        c = [LABEL[m]]
        for r in REGIMES:
            v, sd = agg[("sampled", m, r)]
            s = f"{fmt(v)} $\\pm$ {fmt(sd)}"
            if r != "R0" and m != "Random":
                s += star(regime_p("sampled", m, r))
            c.append(s)
        if m in BASELINES:
            c += [fmt(bl_s["R0"]["methods"][m]["Hits@10"]["mean"]), fmt(bl_s["R0"]["methods"][m]["AUROC"]["mean"])]
        else:
            k = kge_sum[f"{m}|R0|d64|e300"]
            c += [fmt(k["Hits@10_mean"]), fmt(k["AUROC_type_mean"])]
        rows.append(" & ".join(c) + r" \\")
        if m == "PreferentialAttachment":
            rows.append(r"\midrule")
    write("tableS1_sampled_grid.tex", r"""\begin{tabular}{l cccc cc}
\toprule
Method & MRR R0 & MRR R1 & MRR R2 & MRR R3 & Hits@10 R0 & AUROC R0 \\
\midrule
""" + "\n".join(rows) + "\n" + r"""\bottomrule
\end{tabular}""")

    # ---------------------------------------------------------------- S5 Table
    keep = ~rec
    N["n_recreated"] = int(rec.sum())
    rows = []
    for m in METHODS[1:]:
        c = [LABEL[m]]
        for proto in ("full", "sampled"):
            r0, r2 = rr[(proto, m, "R0")], rr[(proto, m, "R2")]
            n_cand = POOL["R0"] if proto == "full" else 51
            mrand = full_random_mrr(n_cand)
            all_ = mrr(r2) / mrr(r0)
            cc = (mrr(r2) - mrand) / (mrr(r0) - mrand)
            ex_same = mrr(r2, keep) / mrr(r0, keep)
            ex_mixed = mrr(r2, keep) / mrr(r0)
            c += [f"{100*all_:.1f}\\%", f"{100*cc:.1f}\\%", f"{100*ex_same:.1f}\\%", f"{100*ex_mixed:.1f}\\%"]
            N.setdefault("S5", {})[f"{proto}|{m}"] = dict(all=all_, chance=cc, excl_same=ex_same,
                                                         excl_mixed=ex_mixed,
                                                         rec_contrib=1 - mrr(r2, keep) / mrr(r2))
        rows.append(" & ".join(c) + r" \\")
        if m == "PreferentialAttachment":
            rows.append(r"\midrule")
    write("tableS5_retained_variants.tex", r"""\begin{tabular}{l cccc cccc}
\toprule
 & \multicolumn{4}{c}{Full ranking} & \multicolumn{4}{c}{Sampled ranking} \\
\cmidrule(lr){2-5}\cmidrule(lr){6-9}
Method & Retained & Chance-corr. & Excl.\ rec.$^{a}$ & Excl.\ rec.$^{b}$ & Retained & Chance-corr. & Excl.\ rec.$^{a}$ & Excl.\ rec.$^{b}$ \\
\midrule
""" + "\n".join(rows) + "\n" + r"""\bottomrule
\end{tabular}""")

    # ---------------------------------------------------------------- S6 Table
    blocks = {"A regime effect": "A. Each regime against R0 (48 comparisons)",
              "B R2 vs PA": "B. Each method's R2 score against PA's R2 score (14 comparisons)",
              "C method contrast": "C. Adamic--Adar against each other method at R0 (14 comparisons)"}
    rows = []
    for bkey, btitle in blocks.items():
        rows.append(r"\midrule" + "\n" + rf"\multicolumn{{7}}{{l}}{{\textit{{{btitle}}}}} \\" + "\n" + r"\midrule")
        for proto in ("full", "sampled"):
            for t in [t for t in mt["tests"] if t["block"].startswith(bkey[:1]) and t["protocol"] == proto]:
                comp = t["comparison"].replace("Adamic-Adar", "Adamic--Adar").replace(" vs ", " vs.\\ ")
                rows.append(" & ".join([
                    comp, "Full" if proto == "full" else "Sampled", fmt(t["mrr_a"]), fmt(t["mrr_b"]),
                    f"{signed(t['delta'])} [{signed(t['ci_low'])}, {signed(t['ci_high'])}]",
                    f"{t['p']:.1e}".replace("e-0", "e$-$").replace("e+00", ""),
                    pfmt(t["p_holm"]) + ("" if t["significant"] else " (n.s.)")]) + r" \\")
    write("tableS6_pvalues_grouped.tex", r"""\begin{longtable}{l l c c l c l}
\toprule
Comparison & Protocol & MRR (first) & MRR (second) & Difference [95\% CI] & $p$ & $p_{\mathrm{Holm}}$ \\
""" + "\n".join(rows) + "\n" + r"""\bottomrule
\end{longtable}""")

    # ================================================================ FIGURES
    fig1_schematic()
    fig2_metrics(agg, bl_s, kge_sum)
    fig3_regimes(agg, regime_p)
    fig4_r2_vs_pa(agg, rshare, pa_test)
    fig5_quartiles(strat, drng, grng)
    cap = RES / "hub_cap" / "hub_cap_analysis.json"
    if cap.exists():  # written by scripts/hub_cap_analysis.py
        figS4_hub_cap(load(cap))
    figS3_gnn_probe(agg)

    (OUT / "revision_numbers.json").write_text(json.dumps(N, indent=1, default=float), encoding="utf-8")
    print("  wrote revision_numbers.json")


# --------------------------------------------------------------------------- Fig 1
def fig1_schematic():
    pos = {"G1": (0, 2), "G2": (0, 1), "G3": (0, 0), "D1": (2.2, 2), "D2": (2.2, 1), "D3": (2.2, 0),
           "P": (1.1, 2.85), "M": (1.1, -0.85)}
    kind = {"G1": "gene", "G2": "gene", "G3": "gene", "D1": "dis", "D2": "dis", "D3": "dis",
            "P": "phen", "M": "mouse"}
    label = {"G1": "$g_1$", "G2": "$g_2$", "G3": "$g_3$", "D1": "$d_1$", "D2": "$d_2$", "D3": "$d_3$",
             "P": "$p$", "M": "$g_m$"}
    gd = [("G1", "D2"), ("G2", "D2"), ("G3", "D3")]           # gene-disease training edges
    other = [("G1", "D1"), ("G2", "D3")]                      # another gene-disease relation
    phen = [("G1", "P"), ("D1", "P")]                         # phenotype edges
    orth = [("G3", "M")]                                      # orthology
    model = [("M", "D1")]                                     # MODEL_OF
    test = ("G1", "D1")
    regimes = {
        "R0  unmodified": dict(gd=gd, other=other, phen=phen, orth=orth, model=model, removed=[]),
        "R1  direct overlap removed": dict(gd=gd, other=[other[1]], phen=phen, orth=orth, model=model,
                                           removed=[("other", other[0])]),
        "R2  degree-preserving rewiring": dict(gd=[("G1", "D3"), ("G2", "D2"), ("G3", "D2")][:0] +
                                               [("G1", "D2"), ("G2", "D3"), ("G3", "D2")],
                                               other=[("G1", "D3"), ("G2", "D1")], phen=phen,
                                               orth=orth, model=model, removed=[]),
        "R3  cross-species edges removed": dict(gd=gd, other=other, phen=phen, orth=[], model=[],
                                                removed=[("orth", orth[0]), ("model", model[0])]),
    }
    # R2 must keep every node's per-relation degree; check it rather than trust the drawing
    def degs(edges):
        c = {}
        for a, b in edges:
            c[a] = c.get(a, 0) + 1
            c[b] = c.get(b, 0) + 1
        return c
    for rel in ("gd", "other"):
        assert degs(regimes["R0  unmodified"][rel]) == degs(regimes["R2  degree-preserving rewiring"][rel]), rel
    sty = {"gd": dict(color="#333333", lw=1.6, ls="-"),
           "other": dict(color="#0072B2", lw=1.6, ls="-"),
           "phen": dict(color="#999999", lw=1.1, ls="-"),
           "orth": dict(color="#D55E00", lw=1.6, ls="-"),
           "model": dict(color="#D55E00", lw=1.6, ls=(0, (1, 1.2)))}
    fig, axes = plt.subplots(1, 4, figsize=(10.2, 3.3))
    for ax, (title, R) in zip(axes, regimes.items()):
        ax.set_axis_off()
        ax.set_xlim(-0.6, 2.8)
        ax.set_ylim(-1.35, 3.35)
        head, rest = title.split("  ", 1)
        ax.set_title(f"{head}\n{rest}", fontsize=8.5, loc="left", fontweight="bold")
        for rel in ("phen", "gd", "other", "orth", "model"):
            for a, b in R[rel]:
                (x1, y1), (x2, y2) = pos[a], pos[b]
                rad = 0.25 if rel == "other" else 0.0
                ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                            arrowprops=dict(arrowstyle="-", connectionstyle=f"arc3,rad={rad}",
                                            shrinkA=9, shrinkB=9, **sty[rel]))
        for rel, (a, b) in R["removed"]:
            (x1, y1), (x2, y2) = pos[a], pos[b]
            rad = 0.25 if rel == "other" else 0.0
            ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle="-", connectionstyle=f"arc3,rad={rad}", shrinkA=9,
                                        shrinkB=9, color="#BBBBBB", lw=1.0, ls=(0, (3, 2))))
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2 - (0.27 if rel == "other" else 0)
            ax.text(mx, my, "×", color="#C00000", fontsize=13, ha="center", va="center", fontweight="bold")
        (x1, y1), (x2, y2) = pos[test[0]], pos[test[1]]
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-", connectionstyle="arc3,rad=-0.25", shrinkA=9, shrinkB=9,
                                    color="#C00000", lw=1.4, ls=(0, (4, 2))))
        for n, (x, y) in pos.items():
            k = kind[n]
            fc = {"gene": "#FFFFFF", "dis": "#FFFFFF", "phen": "#EEEEEE", "mouse": "#FCE6D8"}[k]
            mk = {"gene": "o", "dis": "s", "phen": "D", "mouse": "o"}[k]
            ax.scatter([x], [y], s=330, marker=mk, facecolor=fc, edgecolor="#222222", lw=1.1, zorder=3)
            ax.text(x, y, label[n], ha="center", va="center", fontsize=8.5, zorder=4)
        if title.startswith("R2"):
            ax.text(2.3, -0.95, "every node keeps\nits degree in\neach relation", ha="center", va="center",
                    fontsize=7, color="#444444")
    handles = [Line2D([], [], **sty["gd"], label="gene–disease (target relation)"),
               Line2D([], [], **sty["other"], label="other gene–disease relation"),
               Line2D([], [], **sty["phen"], label="phenotype relation"),
               Line2D([], [], **sty["orth"], label="orthology"),
               Line2D([], [], **sty["model"], label="MODEL_OF"),
               Line2D([], [], color="#C00000", lw=1.4, ls=(0, (4, 2)), label="held-out test pair $g_1$–$d_1$"),
               Line2D([], [], marker="o", ls="", mfc="w", mec="k", label="human gene"),
               Line2D([], [], marker="s", ls="", mfc="w", mec="k", label="disease"),
               Line2D([], [], marker="o", ls="", mfc="#FCE6D8", mec="k", label="non-human gene")]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=7, bbox_to_anchor=(0.5, -0.1))
    savefig(fig, "fig1_regime_schematic")


# --------------------------------------------------------------------------- Fig 2
def fig2_metrics(agg, bl_s, kge_sum):
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.2, 3.8))
    for m in METHODS:
        if m in BASELINES:
            au = bl_s["R0"]["methods"][m]["AUROC"]["mean"]
        else:
            au = kge_sum[f"{m}|R0|d64|e300"]["AUROC_type_mean"]
        ms, mf = agg[("sampled", m, "R0")][0], agg[("full", m, "R0")][0]
        for ax, x, y in ((a, au, ms), (b, ms, mf)):
            ax.scatter(x, y, s=55, marker=MARKER[m], color=COLOR[m], edgecolor="k", lw=0.4, zorder=3,
                       label=PLAIN[m])
    a.set_xlabel("AUROC (sampled, type-matched negatives)")
    a.set_ylabel("MRR, sampled ranking (51 candidates)")
    a.set_title("A  AUROC vs sampled MRR at R0", loc="left", fontweight="bold")
    b.set_xlabel("MRR, sampled ranking (51 candidates)")
    b.set_ylabel("MRR, full ranking (14,307 candidates)")
    b.set_title("B  Sampled vs full-ranking MRR at R0", loc="left", fontweight="bold")
    a.set_xlim(0.45, 1.0)
    a.set_ylim(0, 0.9)
    b.set_xlim(0, 0.9)
    b.set_ylim(0, 0.15)
    a.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    savefig(fig, "fig2_metrics_protocols")


# --------------------------------------------------------------------------- Fig 3
def fig3_regimes(agg, regime_p):
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.4))
    x = np.arange(4)
    for row, proto in enumerate(("full", "sampled")):
        for col, group in enumerate((BASELINES, KGE)):
            ax = axes[row, col]
            ax.axvspan(1.6, 2.4, color="#F1F1F1", zorder=0)
            for m in group:
                v = np.array([agg[(proto, m, r)][0] for r in REGIMES])
                sd = np.array([agg[(proto, m, r)][1] for r in REGIMES])
                ax.plot(x, v, color=COLOR[m], lw=1.3, zorder=2)
                if sd.any():
                    ax.fill_between(x, v - sd, v + sd, color=COLOR[m], alpha=0.18, lw=0)
                for i, r in enumerate(REGIMES):
                    p = None if (r == "R0" or m == "Random") else regime_p(proto, m, r)
                    filled = r == "R0" or (p is not None and p < 0.05)
                    ax.scatter(x[i], v[i], marker=MARKER[m], s=40, zorder=3, lw=0.9,
                               facecolor=COLOR[m] if filled else "white", edgecolor=COLOR[m])
            ax.set_xticks(x, ["R0\nunmodified", "R1\ndirect overlap\nremoved", "R2\nrewired",
                              "R3\ncross-species\nremoved"], fontsize=7)
            ax.set_ylabel("MRR, " + ("full ranking" if proto == "full" else "sampled ranking"))
            ax.set_ylim(bottom=0)
            ttl = ("Topological baselines" if col == 0 else "KGE models") + \
                  (", full ranking" if proto == "full" else ", sampled ranking")
            ax.set_title("ABCD"[row * 2 + col] + "  " + ttl, loc="left", fontweight="bold", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.legend(handles=[Line2D([], [], color=COLOR[m], marker=MARKER[m], label=PLAIN[m]) for m in METHODS],
               fontsize=7, ncol=9, loc="lower center", bbox_to_anchor=(0.5, 0.025))
    fig.text(0.5, 0.0, "Filled marker: differs from R0 after Holm correction (p$_{Holm}$ < 0.05); "
             "open marker: not significant. Shaded column: R2.", ha="center", fontsize=7.5)
    savefig(fig, "fig3_regimes")


# --------------------------------------------------------------------------- Fig 4
def fig4_r2_vs_pa(agg, rshare, pa_test):
    ms = ["CommonNeighbors", "AdamicAdar", "Jaccard", "TransE", "RotatE", "DistMult", "ComplEx",
          "PreferentialAttachment"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))
    for ax, proto, ttl in ((axes[0], "full", "A  Full ranking"), (axes[1], "sampled", "B  Sampled ranking")):
        y = np.arange(len(ms))[::-1]
        for yi, m in zip(y, ms):
            r0, r2 = agg[(proto, m, "R0")][0], agg[(proto, m, "R2")][0]
            ax.barh(yi, r0, color=COLOR[m], alpha=0.28, edgecolor=COLOR[m], height=0.7)
            ax.barh(yi, r2, color=COLOR[m], height=0.7)
            rs = rshare[(proto, m)]
            txt = f"{100*rs['retained_share']:.0f}% ({100*rs['ci_low']:.0f}–{100*rs['ci_high']:.0f})"
            if m != "PreferentialAttachment":
                t = pa_test(proto, m)
                txt += "" if t["significant"] else "  ≈PA"
            ax.text(max(r0, r2) * 1.02 + 0.002, yi, txt, va="center", fontsize=6.8)
        pa2 = agg[(proto, "PreferentialAttachment", "R2")][0]
        ax.axvline(pa2, color=COLOR["PreferentialAttachment"], ls="--", lw=1.1)
        ax.text(pa2, len(ms) - 0.35, " PA at R2", color="#9A6700", fontsize=7, va="bottom")
        ax.set_yticks(y, [PLAIN[m] for m in ms])
        ax.set_xlabel("MRR" + (" (14,307 candidates)" if proto == "full" else " (51 candidates)"))
        ax.set_title(ttl, loc="left", fontweight="bold")
        ax.set_xlim(0, (0.175 if proto == "full" else 1.12))
        ax.grid(axis="y", visible=False)
    axes[0].legend(handles=[Patch(color="#888888", alpha=0.3, label="R0 MRR"),
                            Patch(color="#555555", label="R2 MRR (retained)")],
                   loc="lower right", fontsize=7)
    fig.tight_layout()
    savefig(fig, "fig4_retained_vs_pa")


# --------------------------------------------------------------------------- Fig 5
def fig5_quartiles(strat, drng, grng):
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.8))
    q = np.arange(1, 5)
    for row, proto in enumerate(("full", "sampled")):
        for col, key in enumerate(("gQ", "dQ")):
            ax = axes[row, col]
            for m in METHODS:
                if (proto, m, "R0") not in strat:
                    continue
                ax.plot(q, strat[(proto, m, "R0")][key], marker=MARKER[m], color=COLOR[m], lw=1.2, ms=5,
                        label=PLAIN[m])
            rng = grng if key == "gQ" else drng
            ax.set_xticks(q, [f"Q{i}\n{a:,}–{b:,}" for i, (a, b) in enumerate(rng, 1)], fontsize=7)
            ax.set_ylabel("R0 MRR, " + ("full ranking" if proto == "full" else "sampled ranking"))
            ax.set_ylim(bottom=0)
            ttl = ("Query-gene degree" if key == "gQ" else "True-disease degree") + \
                  (", full ranking" if proto == "full" else ", sampled ranking")
            ax.set_title("ABCD"[row * 2 + col] + "  " + ttl, loc="left", fontweight="bold", fontsize=8.5)
            ax.set_xlabel(("Query-gene" if key == "gQ" else "True-disease")
                          + " degree quartile (R0 training degree)", fontsize=7.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.legend(handles=[Line2D([], [], color=COLOR[m], marker=MARKER[m], label=PLAIN[m]) for m in METHODS],
               fontsize=7, ncol=9, loc="lower center", bbox_to_anchor=(0.5, 0.0))
    savefig(fig, "fig5_degree_quartiles")


# --------------------------------------------------------------------------- S3 Fig
def figS3_gnn_probe(agg):
    """Same data as the first submission's S3 Fig, with plain titles (reviewer MC7)."""
    import glob
    PLOS_NAME["figS3_gnn_probe"] = "S3_Fig"
    strat = load(RES / "gnn" / "degree_stratified.json")["by_disease_degree"]
    res, rng = strat["results"], strat["strata"]["quartiles"]
    delta = load(RES / "gnn" / "gnn_regime_delta.json")["models"]["DistMult"]
    full_r0 = agg[("sampled", "DistMult", "R0")][0]
    full_drop = 1 - agg[("sampled", "DistMult", "R2")][0] / full_r0
    pts = [(0.05, -delta["pct_change"] / 100)]
    for p in sorted(glob.glob(str(RES / "gnn_refcal" / "*" / "kge_summary.json"))):
        a = load(Path(p))
        frac = float(a["runs"][0]["subgraph_frac"])
        r0 = a["aggregate"]["DistMult|R0|d64|e300"]["MRR_mean"]
        r2 = a["aggregate"]["DistMult|R2|d64|e300"]["MRR_mean"]
        pts.append((frac, 1 - r2 / r0))
    pts.append((1.0, full_drop))
    pts.sort()
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.2, 3.8))
    q = np.arange(1, 5)
    for key, lab, col, mk in (("RGCN|R0", "R-GCN", "#0072B2", "o"),
                              ("DistMult|R0", "DistMult control", "#D55E00", "s")):
        m = np.array([res[key]["quartiles"][f"Q{i}"]["MRR_mean"] for i in q])
        s = np.array([res[key]["quartiles"][f"Q{i}"]["MRR_sd"] for i in q])
        h = res[key]["halves"]
        a.plot(q, m, marker=mk, color=col, lw=1.4,
               label=f"{lab} (upper/lower half: {h['high']['MRR_mean'] / h['low']['MRR_mean']:.2f}×)")
        a.fill_between(q, m - s, m + s, color=col, alpha=0.18, lw=0)
    a.set_xticks(q, [f"Q{i}\n{rng[f'Q{i}']['degree_min']:,}–{rng[f'Q{i}']['degree_max']:,}" for i in q],
                 fontsize=7)
    a.set_xlabel("True-disease degree quartile (full-graph R0 training degree)")
    a.set_ylabel("R0 MRR, sampled ranking (5% subgraph)")
    a.set_ylim(0, 1)
    a.legend(fontsize=7, loc="upper left")
    a.set_title("A  R0 MRR by disease degree", loc="left", fontweight="bold")
    fr = [p[0] for p in pts]
    sh = [p[1] / full_drop for p in pts]
    b.plot(fr, sh, color="#555555", lw=1.2, zorder=1)
    for f, v in zip(fr, sh):
        ok = v >= 0.5
        b.scatter(f, v, s=55, zorder=2, marker="o" if ok else "X",
                  color="#009E73" if ok else "#D55E00")
        b.text(f, v + 0.05, f"{100 * v:.0f}%", ha="center", fontsize=7)
    b.axhline(0.5, color="#009E73", ls="--", lw=1)
    b.text(0.052, 0.53, "validity threshold (50%), fixed before the run", fontsize=7, color="#007A5A")
    b.set_xscale("log")
    b.set_xticks(fr, [f"{f:g}" for f in fr])
    b.minorticks_off()
    b.set_ylim(0, 1.12)
    b.set_xlabel("Fraction of training edges in the subgraph (log scale)")
    b.set_ylabel("Share of the full-graph R0-to-R2 drop\nreproduced by the DistMult control")
    b.set_title("B  Subgraph validity check", loc="left", fontweight="bold")
    b.legend(handles=[Line2D([], [], marker="X", ls="", color="#D55E00", label="fails the check"),
                      Line2D([], [], marker="o", ls="", color="#009E73", label="passes the check")],
             fontsize=7, loc="lower right")
    fig.tight_layout()
    savefig(fig, "figS3_gnn_probe")


# --------------------------------------------------------------------------- S4 Fig
def figS4_hub_cap(j):
    PLOS_NAME["figS4_hub_cap"] = "S4_Fig"
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.2, 3.6))
    bands = j["enrichment_by_neighbor_degree"]
    lab = [e["band"].replace("-", "–") for e in bands]
    en = [e["enrichment"] for e in bands]
    x = np.arange(len(bands))
    a.bar(x, en, color=["#0072B2" if v > 2 else "#BBBBBB" for v in en], width=0.7)
    a.axhline(1, color="k", lw=0.8, ls="--")
    a.axvline(4.5, color="#D55E00", lw=1.2)
    a.text(4.55, 300, " cap\n 2,000", color="#D55E00", fontsize=7.5, va="center")
    for xi, v, e in zip(x, en, bands):
        a.text(xi, v * 1.15, f"{v:,.0f}×" if v >= 10 else f"{v:.1f}×", ha="center", fontsize=7)
    a.set_yscale("log")
    a.set_ylim(0.3, 5000)
    a.set_xticks(x, lab, fontsize=7, rotation=30)
    a.set_xlabel("Degree of the shared neighbor")
    a.set_ylabel("Enrichment (true / random pairs)")
    a.set_title("A  Shared-neighbor enrichment by degree", loc="left", fontweight="bold")
    caps = [r["cap"] for r in j["cap_sweep"]]
    xc = np.arange(len(caps))
    for m, key in (("CommonNeighbors", "CommonNeighbors"), ("AdamicAdar", "AdamicAdar")):
        b.plot(xc, [r[key] for r in j["cap_sweep"]], marker=MARKER[m], color=COLOR[m], label=PLAIN[m])
    b.axvline(caps.index(2000), color="#D55E00", lw=1.2)
    b.set_xticks(xc, [f"{c:,}" if c else "none" for c in caps], fontsize=7)
    b.set_xlabel("Neighbor-degree cap")
    b.set_ylabel("Validation MRR (sampled ranking)")
    b.set_title("B  Cap sweep on validation edges", loc="left", fontweight="bold")
    b.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    savefig(fig, "figS4_hub_cap")


if __name__ == "__main__":
    main()
