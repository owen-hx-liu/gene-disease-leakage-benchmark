# Quantifying evaluation leakage in multi-species gene-disease link prediction

A graded leakage audit and a de-leaked benchmark for gene-disease link prediction on a
multi-species, Monarch-derived knowledge graph. This repository holds the code, the
de-leaked evaluation splits, the benchmark results, and the scripts that regenerate every
table and figure from the frozen result files.

> **Scope note.** This is a leakage-aware benchmark, not a database-integration resource.
> The knowledge graph is about 98.3% Monarch Initiative content (Monarch is itself an
> aggregator) augmented with seven curated non-Monarch sources. It is described honestly as
> a Monarch-derived graph, not an equal integration of many independent databases.

## What this is

Knowledge-graph embedding (KGE) models are widely used to predict gene-disease associations
and are usually reported with high ranking and AUROC scores. Those scores can be inflated by
evaluation leakage. This project measures how much, and from which sources, using a graded
set of controlled evaluation regimes run through one shared ranking harness.

Four KGE models spanning two families (translational: TransE, RotatE; bilinear: DistMult,
ComplEx) and five topological baselines (Random, Common-Neighbors, Adamic-Adar, Jaccard,
Preferential-Attachment) are evaluated on 4,228 held-out human gene-disease test edges
across four regimes, with three seeds (42, 1, 7):

| Regime | Name | What it removes |
|---|---|---|
| R0 | standard | nothing (ordinary random split) |
| R1 | redundancy | direct held-pair edges (train copies of test facts) |
| R2 | degree-preserving null | genuine structure, while preserving the degree sequence exactly |
| R3 | orthology-blocked | cross-species ORTHOLOGOUS_TO bridges |

The graph has 452,012 nodes and 5,860,539 directed edges across 28 relation types and 8
node categories.

## Repository layout

```
scripts/                     pipeline and analysis code
  build_deleaked_splits.py   build the R0/R1/R3 splits and the split manifest
  build_degree_null.py       build the R2 degree-preserving null
  run_baselines.py           topological baselines across regimes
  run_kge.py                 TransE, RotatE, DistMult, ComplEx (GPU); --full-rank for the robustness run
  run_robustness.py          baseline full-ranking robustness
  hetionet_audit.py          cross-graph replication on Hetionet
  degree_stratified.py       degree-stratified re-analysis (gene vs disease degree; no retraining)
  case_study.py              worked case study (degree false-positives vs missed rare edges)
  lib_eval.py                the single shared ranking and scoring harness
  make_tables.py             Tables 1-6 and S1 from result JSONs (no hand-typed numbers)
  make_figures.py            Figures 1-4 from result JSONs
data/processed/
  splits/                    the de-leaked benchmark (train/valid/test, R1/R2/R3, manifest)
  results/                   per-regime baseline, KGE, robustness, Hetionet, degree-stratified, and case-study results
  provenance/                per-source contribution table
tables/                      generated tables (Markdown and LaTeX)
figures/                     generated figures (PNG and PDF)
run_all.sh                   one-command end-to-end reproduction
requirements.txt             Python dependencies
```

## Reproducing the results

```bash
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
# GPU (RTX 50-series / Blackwell needs CUDA 12.8+):
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128

./run_all.sh                 # full pipeline (the KGE step needs a CUDA GPU)
./run_all.sh --skip-kge      # everything except KGE, reusing the frozen result JSONs
```

A bare `git clone` only carries a partial subset of `data/processed/results/` (kept as a small,
browsable sample; GitHub has no clean way to track ~9 MB of JSON without also inviting the 160 MB
`splits.zip` past its 100 MB per-file limit). To regenerate every table and figure without a GPU,
first download and unzip `splits.zip` and `results.zip` from the
[Zenodo record](https://doi.org/10.5281/zenodo.21828739) into `data/processed/`, then:

```bash
python scripts/make_tables.py    # tables/table1-6 and tableS1 (.md and .tex)
python scripts/make_figures.py   # figures/fig1-4 (.png and .pdf)
```

All runs use fixed seeds (42, 1, 7). Paths are repo-relative; there are no hard-coded machine
paths or secrets. Heavy steps checkpoint to disk. Reproducing the KGE arm from scratch needs
a CUDA GPU and is slow (about 40 minutes per model-by-regime-by-seed cell); the frozen result
JSONs on Zenodo let you rebuild every table and figure without retraining.

## Data, provenance, and licensing

The knowledge graph is derived from the Monarch Initiative (about 98.3% of edges; CC BY 4.0)
augmented with seven curated sources: DGIdb, GWAS Catalog, DrugCentral, Orphadata,
Gene2Phenotype, CIViC, and DIDA. Per-source contributions are archived as
`provenance/source_provenance.csv` in `provenance.zip` on the
[Zenodo record](https://doi.org/10.5281/zenodo.21828739) (the `data/processed/provenance/`
path is git-ignored, not part of the GitHub tree). Because Monarch aggregates several primary
sources, do not read "supported by N datasets" as N independent votes.

Two licenses apply:

- **Code** (everything in `scripts/`): MIT. See [`LICENSE`](LICENSE).
- **Data** (the de-leaked splits and result files under `data/processed/`): CC BY 4.0, with
  attribution to the Monarch Initiative and the underlying primary sources. See
  [`LICENSE-DATA`](LICENSE-DATA).

## Citation

If you use this benchmark or code, please cite the archived release:
[10.5281/zenodo.21828739](https://doi.org/10.5281/zenodo.21828739). Machine-readable metadata
is in [`CITATION.cff`](CITATION.cff) and [`.zenodo.json`](.zenodo.json).

## Caveats

- **No real temporal split.** The graph carries no genuine edge-discovery dates, so the
  benchmark controls optimism with the leakage regimes R0 to R3 on a random held-out split,
  not a time split.
- **Coverage is Monarch-dominated.** Even after integrating seven more sources, about 98.3%
  of edges are Monarch content.
- **Name the protocol.** Ranking metrics are reported for both a sampled-negative protocol
  (Table 2) and a filtered full-ranking protocol (Table S1); the two can rank methods
  differently, so always state which one a number comes from.
