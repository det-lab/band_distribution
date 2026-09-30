#!/bin/bash
# Build one normalization table end to end for a fit region: measure how many
# nodes each axis needs, evaluate the grid, check it against held-out points.
#
#   sbatch slurm/build_table.job BAND EP_MIN EP_MAX EQ_MIN EQ_MAX
#   sbatch slurm/build_table.job NR 2.5 350 0.75 200
#
# Stages (each Slurm stage is `sbatch --wait slurm/normgrid.sbatch ...`, so
# this script keeps running until the whole chain is done, hence the driver
# job slurm/build_table.job, which runs it with the arguments above):
#   1. study   normplan.py points   -> ~700 integrals along each axis (array)
#   2. plan    normplan.py analyze  -> node count per axis, grid spec, report
#   3. grid    the tensor grid, and HELDOUT random validation points (arrays)
#   4. check   normgrid.py merge + validate; exit status 1 if the worst
#              held-out relative error exceeds ACCEPT
# The light Python steps (plan, merge, validate) run with $PYTHON on the login
# node, as slurm/submit_alderaan.sh does; the integrals run in the container
# via normgrid.sbatch (BAND_SIF etc. are read there; BAND_NATIVE=1 works too).
# Every array is checked with sacct once it ends: `sbatch --wait` exits 0 on
# Alderaan even when tasks fail, so a failed task stops the chain here instead.
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
#   BOX=          replace axes of the band's default box (normgrid.DEFAULT_BOX),
#                 e.g. BOX="V=2.5:4 dq=0:0.42" -- what PriorNotCoveredError
#                 suggests when an MCMC prior reaches outside a table.  A custom
#                 box gets its own build directory and table name (tag
#                 suffix _box<hash>): study and held-out results depend on it.
#   ACCOUNT= PARTITION= TIME=   passed to sbatch when set
#   BAND_SIF=/scratch/$USER/containers/band.sif   the container (as in
#                 normgrid.sbatch); checked here before anything is submitted
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
BOX_ARGS=()
if [[ -n "${BOX:-}" ]]; then
    read -r -a BOX_ARGS <<< "$BOX"
    # the hash of the sorted entries names the box: same box, same directory
    TAG+=_box$(printf '%s\n' "${BOX_ARGS[@]}" | sort | sha1sum | cut -c1-8)
fi
WORK=${WORKDIR:-build/$TAG}
PLAN=$WORK/tol$TOL
OUT_TABLE=${OUT_TABLE:-tables/norm_$TAG.h5}
mkdir -p logs "$WORK/study_results" "$WORK/held_results" "$PLAN/results" "$(dirname "$OUT_TABLE")"

if [[ "${BAND_NATIVE:-0}" != "1" ]]; then
    export BAND_SIF=${BAND_SIF:-/scratch/$USER/containers/band.sif}
    [[ -f "$BAND_SIF" ]] || { echo "no container at $BAND_SIF: set BAND_SIF, or pull one with slurm/pull_container.job" >&2; exit 1; }
fi

SBATCH_ARGS=()
[[ -n "${ACCOUNT:-}" ]] && SBATCH_ARGS+=(--account="$ACCOUNT")
[[ -n "${PARTITION:-}" ]] && SBATCH_ARGS+=(--partition="$PARTITION")
[[ -n "${TIME:-}" ]] && SBATCH_ARGS+=(--time="$TIME")

NG=("$PYTHON" python/normgrid.py)
NP=("$PYTHON" python/normplan.py)
REGION=("$EP_MIN" "$EP_MAX" "$EQ_MIN" "$EQ_MAX")
BOXOPT=(); [[ ${#BOX_ARGS[@]} -gt 0 ]] && BOXOPT=(--box "${BOX_ARGS[@]}")

array() {   # array SPEC CHUNK_SIZE RESULTS_DIR: an sbatch array over the spec's chunks, waited for
    local spec=$1 size=$2 out=$3 n job
    n=$("${NG[@]}" chunks "$spec" --size "$size" | wc -l)
    echo "[$(date +%T)] $(basename "$spec"): $n array tasks of up to $size points"
    # --parsable prints the job id, but Alderaan's sbatch wrapper may print
    # INFO lines first: take the last line that is a job id
    job=$(sbatch --parsable --wait ${SBATCH_ARGS[@]+"${SBATCH_ARGS[@]}"} --array=0-$((n - 1))%"$THROTTLE" \
                 --job-name="ng_${BAND}_$(basename "$spec" .json)" \
                 slurm/normgrid.sbatch "$spec" "$size" "$out" | grep -E '^[0-9]+(;|$)' | tail -1 | cut -d';' -f1 || true)
    [[ -n "$job" ]] || { echo "sbatch did not report a job id for $spec" >&2; return 1; }
    check_tasks "$job" "$n"
}

check_tasks() {   # check_tasks JOB N: fail unless all N tasks of array JOB are COMPLETED
    local job=$1 n=$2 states bad try
    # the accounting database can lag the end of the job by a few seconds
    for try in 1 2 3 4 5 6; do
        states=$(sacct -j "$job" -X -n -P -o JobID,State,ExitCode)
        [[ $(grep -c . <<< "$states") -ge $n ]] && ! grep -qE '\|(PENDING|RUNNING|REQUEUED|COMPLETING)' <<< "$states" && break
        sleep 10
    done
    bad=$(awk -F'|' '$2 != "COMPLETED"' <<< "$states")
    if [[ $(grep -c . <<< "$states") -lt $n || -n "$bad" ]]; then
        echo "array job $job: not every one of its $n tasks COMPLETED; see logs/normgrid_${job}_*.err" >&2
        head -5 <<< "${bad:-$states}" >&2
        return 1
    fi
    echo "[$(date +%T)] array job $job: all $n tasks COMPLETED"
}

echo "== 1. study: how many nodes does each axis need? ($TAG)"
"${NP[@]}" points --band "$BAND" --region "${REGION[@]}" --epsrel "$EPSREL" ${BOXOPT[@]+"${BOXOPT[@]}"} \
    --out "$WORK/study.json"
[[ -n "${BOX:-}" ]] && echo "   box: $BOX"
array "$WORK/study.json" "$STUDY_CHUNK" "$WORK/study_results"

echo "== 2. plan: node counts for tol $TOL"
"${NP[@]}" analyze --study "$WORK/study.json" --results "$WORK/study_results/res_study_*.txt" \
    --tol "$TOL" --max-points "$MAX_POINTS" --out-spec "$PLAN/spec.json" --report "$PLAN/plan.json"

echo "== 3. grid and held-out points"
"${NG[@]}" make-random-spec --band "$BAND" --n "$HELDOUT" --region "${REGION[@]}" --epsrel "$EPSREL" \
    ${BOXOPT[@]+"${BOXOPT[@]}"} \
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
    echo "   TOL=$NEW_TOL sbatch slurm/build_table.job $BAND $EP_MIN $EP_MAX $EQ_MIN $EQ_MAX" >&2
    echo "   (the study and held-out points are reused)" >&2
    exit 1
fi
