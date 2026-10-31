#!/usr/bin/env bash
# =============================================================================
# run_all.sh -- reproduce the leakage-audit benchmark end to end (Unit 12).
#
# Runs, in order:
#   0. hash-verify the input artifacts against ARTIFACT_HASHES.txt  (ABORT on mismatch)
#   1. graph_stats.py            -> data/processed/graph_stats.json
#   2. graph_stats_hetionet.py   -> data/processed/graph_stats_hetionet.json
#   3. build_deleaked_splits.py  -> data/processed/splits/*.csv + split_manifest.json
#   4. build_degree_null.py      -> data/processed/results/null/degree_null.json (+ R2 split; 200 graphs)
#   5. run_baselines.py R0-R3    -> data/processed/results/baselines/baselines_R*.json   (sampled ranking)
#   6. run_robustness.py R0-R3   -> data/processed/results/robustness/robustness_R*.json (full ranking)
#   7. run_kge.py (GPU)          -> data/processed/results/kge/*.json + kge_summary.json  (sampled ranking)
#   8. run_kge.py --full-rank    -> data/processed/results/kge_fullrank/*.json            (full ranking, GPU)
#   9. hetionet_audit.py R0-R2   -> data/processed/results/hetionet/baselines_R*.json
#  10. degree_stratified.py      -> data/processed/results/degree_stratified/degree_stratified.json
#  11. case_study.py             -> data/processed/results/case_study/case_study.json
#  12. make_figures.py           -> figures/fig1-4.{png,pdf}    (first-submission figures)
#  13. make_tables.py            -> tables/table1-6.{md,tex}    (first-submission tables)
#  14. multiple_testing.py       -> data/processed/results/stats/multiple_testing.json + tables/tableS6_pvalues.*
#  15. hub_cap_analysis.py       -> data/processed/results/hub_cap/hub_cap_analysis.json
#  16. make_revision_assets.py   -> revision/tables, revision/figures, revision/revision_numbers.json
#  17. overlap_lowdegree.py      -> revision/tables/tableS8_overlap_lowdegree.tex
#  18. pool_exclusion.py         -> revision/tables/tableS9_pool_exclusion.tex
#  19. hash-verify reproduced splits (informational)
#
# Steps 14-18 build every table and figure of the revised manuscript (full ranking
# primary) from the frozen result JSONs; they need no GPU.
# Acceptance: a clean clone + the data channel reproduces Tables 3-4 and Fig 3-4.
# Runtimes below are approximate on a modern workstation; the KGE step needs a
# CUDA GPU and dominates the wall-clock. Paths are repo-relative; run from anywhere.
#
# Usage:
#   ./run_all.sh                 # full pipeline (KGE included -- needs a GPU)
#   ./run_all.sh --skip-kge      # everything except KGE (reuses committed kge_summary.json)
#   ./run_all.sh --verify-only   # only run the hash checks, then exit
#   ./run_all.sh --skip-hash-check   # skip integrity check (NOT recommended)
#   PYTHON=venv/bin/python ./run_all.sh   # choose the interpreter
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
SKIP_KGE=0
SKIP_HASH=0
VERIFY_ONLY=0

usage() { sed -n '2,39p' "$0"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --skip-kge)        SKIP_KGE=1 ;;
    --skip-hash-check) SKIP_HASH=1 ;;
    --verify-only)     VERIFY_ONLY=1 ;;
    -h|--help)         usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage; exit 2 ;;
  esac
  shift
done

STEP=0
banner() {
  echo
  echo "=============================================================================="
  echo "  $1"
  echo "=============================================================================="
}

# run_py "<title>" "<expected runtime>" "<output location>" <command...>
run_py() {
  local title="$1" est="$2" out="$3"; shift 3
  STEP=$((STEP + 1))
  banner "STEP ${STEP}: ${title}"
  echo "  expected runtime : ${est}"
  echo "  writes           : ${out}"
  echo "  command          : $*"
  echo
  "$@"
  echo
  echo "  [ok] step ${STEP} done -> ${out}"
}

echo "run_all.sh  |  repo: ${ROOT}"
echo "interpreter : ${PYTHON}  ($("$PYTHON" --version 2>&1))"

# ---------------------------------------------------------------- 0. hash-verify
banner "PRE-FLIGHT: verify input artifacts against ARTIFACT_HASHES.txt"
if [ "$SKIP_HASH" -eq 1 ]; then
  echo "  --skip-hash-check set: SKIPPING integrity check (NOT recommended)."
else
  "$PYTHON" scripts/verify_hashes.py --require \
      data/processed/edges_clean_integrated.csv \
      data/external/hetionet/hetionet-v1.0-edges.sif \
      data/external/hetionet/hetionet-v1.0-nodes.tsv
fi

if [ "$VERIFY_ONLY" -eq 1 ]; then
  banner "VERIFY-ONLY: checking every ARTIFACT_HASHES.txt entry present on disk"
  "$PYTHON" scripts/verify_hashes.py --check-existing || true
  exit 0
fi

# --------------------------------------------------------------------- pipeline
run_py "Graph statistics (Monarch)" "~1-2 min" \
  "data/processed/graph_stats.json" \
  "$PYTHON" scripts/graph_stats.py

run_py "Graph statistics (Hetionet robustness graph)" "<1 min" \
  "data/processed/graph_stats_hetionet.json" \
  "$PYTHON" scripts/graph_stats_hetionet.py

run_py "Build de-leaked splits R0/R1/R3 (seed 42)" "~5-15 min" \
  "data/processed/splits/*.csv + split_manifest.json" \
  "$PYTHON" scripts/build_deleaked_splits.py \
      --edges data/processed/edges_clean_integrated.csv \
      --out data/processed/splits --seed 42

run_py "Build R2 degree-preserving null (200 replicates, seed 42)" "hours (200 rewired graphs; longest CPU step)" \
  "data/processed/results/null/degree_null.json + splits/train_R2_degree_null_seed42.csv" \
  "$PYTHON" scripts/build_degree_null.py \
      --splits-dir data/processed/splits --replicates 200 --seed 42

run_py "Topological baselines across regimes R0,R1,R2,R3 (seeds 42,1,7)" "~3-8 min" \
  "data/processed/results/baselines/baselines_R{0,1,2,3}.json" \
  "$PYTHON" scripts/run_baselines.py \
      --regimes R0,R1,R2,R3 --seeds 42,1,7 --splits-dir data/processed/splits

run_py "Topological baselines, full ranking R0,R1,R2,R3 (brute-force check on 150 edges)" "~5 min" \
  "data/processed/results/robustness/robustness_R{0,1,2,3}.json" \
  "$PYTHON" scripts/run_robustness.py --regimes R0,R1,R2,R3 --validate 150

# --------------------------------------------------------------------- 7-8. KGE
if [ "$SKIP_KGE" -eq 1 ]; then
  STEP=$((STEP + 1))
  banner "STEP ${STEP}: KGE (TransE, RotatE, DistMult, ComplEx)  --  SKIPPED (--skip-kge)"
  echo "  Reusing the frozen data/processed/results/kge/ and kge_fullrank/ JSONs."
  echo "  To (re)compute on a CUDA GPU (~40 min/cell, 48 cells per protocol -> many hours):"
  echo "    $PYTHON scripts/run_kge.py --models TransE RotatE DistMult ComplEx \\"
  echo "        --regimes R0 R1 R2 R3 --seeds 42 1 7 --dim 64 --epochs 300"
  echo "    (and the same with --full-rank, one regime per call, for the full-ranking arm)"
  STEP=$((STEP + 1))
else
  run_py "KGE full benchmark: TransE, RotatE, DistMult, ComplEx x R0,R1,R2,R3 x 3 seeds" \
    "GPU REQUIRED; ~40 min/cell, 48 cells -> many hours" \
    "data/processed/results/kge/*.json + kge_summary.json" \
    "$PYTHON" scripts/run_kge.py --models TransE RotatE DistMult ComplEx --regimes R0 R1 R2 R3 \
        --seeds 42 1 7 --dim 64 --epochs 300

  # Full ranking: one call per regime keeps one 5.8M-edge graph in RAM at a time (see
  # scripts/run_fullrank_kgearm.sh). The first regime brute-force-checks the vectorized
  # full-pool rank: 150 edges for the translational models, 30 for the slower bilinear ones.
  STEP=$((STEP + 1))
  banner "STEP ${STEP}: KGE full ranking: 4 models x R0,R1,R2,R3 x 3 seeds (GPU; many hours)"
  for R in R0 R1 R2 R3; do
    VT=0; VB=0
    if [ "$R" = "R0" ]; then VT=150; VB=30; fi
    "$PYTHON" scripts/run_kge.py --full-rank --models TransE RotatE --regimes "$R" \
        --seeds 42 1 7 --dim 64 --epochs 300 --validate-fullrank "$VT"
    "$PYTHON" scripts/run_kge.py --full-rank --models DistMult ComplEx --regimes "$R" \
        --seeds 42 1 7 --dim 64 --epochs 300 --validate-fullrank "$VB"
  done
  "$PYTHON" scripts/run_kge.py --full-rank --merge-only
  echo "  [ok] step ${STEP} done -> data/processed/results/kge_fullrank/"
fi

run_py "Hetionet cross-graph robustness (DaG; R0,R1,R2)" "~2-5 min" \
  "data/processed/results/hetionet/baselines_R{0,1,2}.json" \
  "$PYTHON" scripts/hetionet_audit.py --regimes R0,R1,R2 --seeds 42,1,7

# ------------------------------------------- 8. degree-stratified re-analysis
run_py "Degree-stratified re-analysis (gene vs disease degree; no retraining)" "<1 min" \
  "data/processed/results/degree_stratified/degree_stratified.json" \
  "$PYTHON" scripts/degree_stratified.py

# ------------------------------------------------- 9. worked case study
run_py "Worked case study (degree false-positives vs missed rare edges)" "<1 min" \
  "data/processed/results/case_study/case_study.json" \
  "$PYTHON" scripts/case_study.py

# ---------------------------------------------------------------- 10-11. deliverables
run_py "Manuscript figures (Fig 1-4, png + pdf)" "<1 min" \
  "figures/fig1-4.{png,pdf}" \
  "$PYTHON" scripts/make_figures.py

run_py "Manuscript tables (Tables 1-6 + S1, md + tex)" "<1 min" \
  "tables/table1-6.{md,tex}" \
  "$PYTHON" scripts/make_tables.py

# ------------------------------------------------ 14-18. revised manuscript
run_py "Holm-corrected paired bootstrap, 76 comparisons (100,000 resamples)" "~5-15 min" \
  "data/processed/results/stats/multiple_testing.json + tables/tableS6_pvalues.{md,tex}" \
  "$PYTHON" scripts/multiple_testing.py

run_py "Neighbor-degree cap check for CN/AA (S1 Text, S4 Fig)" "<1 min" \
  "data/processed/results/hub_cap/hub_cap_analysis.json" \
  "$PYTHON" scripts/hub_cap_analysis.py

run_py "Revised manuscript tables and figures (full ranking primary)" "~1-2 min" \
  "revision/tables/, revision/figures/, revision/revision_numbers.json" \
  "$PYTHON" scripts/make_revision_assets.py

run_py "Direct overlap among low-degree diseases (S8 Table)" "<1 min" \
  "revision/tables/tableS8_overlap_lowdegree.tex" \
  "$PYTHON" scripts/overlap_lowdegree.py

run_py "R1/R3 with and without edges whose disease leaves the pool (S9 Table)" "~1-2 min" \
  "revision/tables/tableS9_pool_exclusion.tex" \
  "$PYTHON" scripts/pool_exclusion.py

# ------------------------------------------------------- 10. verify reproduction
banner "POST-RUN: verify reproduced artifacts against ARTIFACT_HASHES.txt"
if "$PYTHON" scripts/verify_hashes.py --check-existing; then
  echo "  [ok] reproduced splits are byte-identical to the frozen manifest."
else
  echo "  [warn] some reproduced artifacts did not match the manifest -- see above."
fi

banner "DONE. Outputs:"
cat <<'EOF'
  graph stats   : data/processed/graph_stats.json, graph_stats_hetionet.json
  splits        : data/processed/splits/
  degree null   : data/processed/results/null/degree_null.json
  baselines     : data/processed/results/baselines/ (sampled), robustness/ (full)
  KGE           : data/processed/results/kge/ (sampled), kge_fullrank/ (full)
  Hetionet      : data/processed/results/hetionet/
  statistics    : data/processed/results/stats/multiple_testing.json
  figures       : figures/fig1-4.{png,pdf} (first submission), revision/figures/ (revised)
  tables        : tables/table1-6.{md,tex} (first submission), revision/tables/ (revised)
EOF
