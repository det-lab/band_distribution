#!/bin/bash
# Build one normalization table end to end for a fit region: measure how many
# nodes each axis needs, evaluate the grid, check it against held-out points.
#
#   slurm/build_table.sh BAND EP_MIN EP_MAX EQ_MIN EQ_MAX
#   slurm/build_table.sh NR 2.5 350 0.75 200
#
# Stages (each Slurm stage is `sbatch --wait slurm/normgrid.sbatch ...`, so
# this script keeps running until the whole chain is done: run it in tmux/
# screen, or as a long single-core job if your cluster lets jobs call sbatch):
#   1. study   normplan.py points   -> ~700 integrals along each axis (array)
#   2. plan    normplan.py analyze  -> node count per axis, grid spec, report
#   3. grid    the tensor grid, and HELDOUT random validation points (arrays)
#   4. check   normgrid.py merge + validate; exit status 1 if the worst
#              held-out relative error exceeds ACCEPT
# The light Python steps (plan, merge, validate) run with $PYTHON on the login
# node, as slurm/submit_alderaan.sh does; the integrals run in the container
# via normgrid.sbatch (BAND_SIF etc. are read there; BAND_NATIVE=1 works too).
#
# Re-running is cheap and safe: `normgrid run` skips finished points.  If the
# check fails, re-run with a tighter TOL (the message says which); the study
# and held-out results are reused, only the grid is recomputed, in its own
# directory (build/<tag>/tol<TOL>/).
#
# Environment (all optional):
#   TOL=1e-7      per-axis interpolation error target for the node counts
#   EPSREL=1e-7   quadrature tolerance of every integral (also floors TOL)
#   ACCEPT=1e-6   worst held-out relative error that counts as a pass
#   HELDOUT=200   number of held-out validation points
#   STUDY_CHUNK=40 GRID_CHUNK=500 HELD_CHUNK=100   points per array task
#   THROTTLE=200  max concurrent tasks of one array
#   MAX_POINTS=500000   refuse a bigger grid (says which axes are expensive)
#   ACCOUNT= PARTITION= TIME=   passed to sbatch when set
#   WORKDIR=build/<tag>   OUT_TABLE=tables/norm_<tag>.h5   PYTHON=python3

set -euo pipefail
cd "$(dirname "$0")/.."

BAND=${1:?usage: slurm/build_table.sh BAND EP_MIN EP_MAX EQ_MIN EQ_MAX}
[[ $# -eq 5 ]] || { echo "usage: slurm/build_table.sh BAND EP_MIN EP_MAX EQ_MIN EQ_MAX" >&2; exit 2; }
EP_MIN=$2 EP_MAX=$3 EQ_MIN=$4 EQ_MAX=$5

TOL=${TOL:-1e-7}; EPSREL=${EPSREL:-1e-7}; ACCEPT=${ACCEPT:-1e-6}; HELDOUT=${HELDOUT:-200}
STUDY_CHUNK=${STUDY_CHUNK:-40}; GRID_CHUNK=${GRID_CHUNK:-500}; HELD_CHUNK=${HELD_CHUNK:-100}
THROTTLE=${THROTTLE:-200}; MAX_POINTS=${MAX_POINTS:-500000}
PYTHON=${PYTHON:-python3}

TAG=${BAND}_ep${EP_MIN}-${EP_MAX}_eq${EQ_MIN}-${EQ_MAX}
WORK=${WORKDIR:-build/$TAG}
PLAN=$WORK/tol$TOL
OUT_TABLE=${OUT_TABLE:-tables/norm_$TAG.h5}
mkdir -p logs "$WORK/study_results" "$WORK/held_results" "$PLAN/results" "$(dirname "$OUT_TABLE")"

SBATCH_ARGS=()
[[ -n "${ACCOUNT:-}" ]] && SBATCH_ARGS+=(--account="$ACCOUNT")
[[ -n "${PARTITION:-}" ]] && SBATCH_ARGS+=(--partition="$PARTITION")
[[ -n "${TIME:-}" ]] && SBATCH_ARGS+=(--time="$TIME")

NG=("$PYTHON" python/normgrid.py)
NP=("$PYTHON" python/normplan.py)
REGION=("$EP_MIN" "$EP_MAX" "$EQ_MIN" "$EQ_MAX")

array() {   # array SPEC CHUNK_SIZE RESULTS_DIR: an sbatch array over the spec's chunks, waited for
    local spec=$1 size=$2 out=$3 n
    n=$("${NG[@]}" chunks "$spec" --size "$size" | wc -l)
    echo "[$(date +%T)] $(basename "$spec"): $n array tasks of up to $size points"
    sbatch --wait ${SBATCH_ARGS[@]+"${SBATCH_ARGS[@]}"} --array=0-$((n - 1))%"$THROTTLE" \
           --job-name="ng_${BAND}_$(basename "$spec" .json)" \
           slurm/normgrid.sbatch "$spec" "$size" "$out"
}

echo "== 1. study: how many nodes does each axis need? ($TAG)"
"${NP[@]}" points --band "$BAND" --region "${REGION[@]}" --epsrel "$EPSREL" --out "$WORK/study.json"
array "$WORK/study.json" "$STUDY_CHUNK" "$WORK/study_results"

echo "== 2. plan: node counts for tol $TOL"
"${NP[@]}" analyze --study "$WORK/study.json" --results "$WORK/study_results/res_study_*.txt" \
    --tol "$TOL" --max-points "$MAX_POINTS" --out-spec "$PLAN/spec.json" --report "$PLAN/plan.json"

echo "== 3. grid and held-out points"
"${NG[@]}" make-random-spec --band "$BAND" --n "$HELDOUT" --region "${REGION[@]}" --epsrel "$EPSREL" \
    --out "$WORK/held.json"
array "$WORK/held.json" "$HELD_CHUNK" "$WORK/held_results" &
held_pid=$!
array "$PLAN/spec.json" "$GRID_CHUNK" "$PLAN/results"
wait "$held_pid"

echo "== 4. merge and validate"
"${NG[@]}" merge --spec "$PLAN/spec.json" --results "$PLAN/results/res_spec_*.txt" --out "$OUT_TABLE"
if "${NG[@]}" validate --table "$OUT_TABLE" --heldout-spec "$WORK/held.json" \
        --results "$WORK/held_results/res_held_*.txt" --accept "$ACCEPT"; then
    echo "== done: $OUT_TABLE"
else
    NEW_TOL=$("$PYTHON" -c "print(f'{$TOL / 4:.0e}')")
    echo "== FAILED the held-out check (worst error above ACCEPT=$ACCEPT); table kept at $OUT_TABLE" >&2
    echo "   the per-axis study cannot see cross terms; retry with more nodes:" >&2
    echo "   TOL=$NEW_TOL slurm/build_table.sh $BAND $EP_MIN $EP_MAX $EQ_MIN $EQ_MAX" >&2
    echo "   (the study and held-out points are reused)" >&2
    exit 1
fi
