#!/bin/bash
# Slurm job array for building a normalization table (python/cli/normgrid.py).
# One array task = one chunk of grid points, single core, no communication;
# the whole grid is embarrassingly parallel.  A TEMPLATE: adjust the #SBATCH
# lines (partition/account/time) for your cluster.  Locally tested with a
# stand-in evaluator and the container paths below checked against the
# Dockerfiles, but not run on a live cluster.
#
# Usage (from the repository root, after `mkdir -p logs results`):
#
#   python python/cli/normgrid.py make-spec --band NR --region 2 200 4 100 \
#          --epsrel 1e-7 --out spec_NR.json          # nodes: RECOMMENDED_NODES
#   python python/cli/normgrid.py make-random-spec --band NR --n 200 \
#          --region 2 200 4 100 --epsrel 1e-7 --out held_NR.json
#   N=$(python python/cli/normgrid.py chunks spec_NR.json --size 500 | wc -l)
#   sbatch --array=0-$((N-1))%200 slurm/normgrid.sbatch spec_NR.json 500 results
#   # same for the held-out validation points (used by `validate` below):
#   N=$(python python/cli/normgrid.py chunks held_NR.json --size 100 | wc -l)
#   sbatch --array=0-$((N-1)) slurm/normgrid.sbatch held_NR.json 100 results
#
#   # when the arrays are done (quick; run on a login node or interactively):
#   python python/cli/normgrid.py merge --spec spec_NR.json --results 'results/res_spec_NR_*.txt' \
#          --out tables/norm_NR_region2-200_4-100.h5
#   python python/cli/normgrid.py validate --table tables/norm_NR_region2-200_4-100.h5 \
#          --heldout-spec held_NR.json --results 'results/res_held_NR_*.txt'
#
# Arguments: SPEC.json  [CHUNK_SIZE=500]  [RESULTS_DIR=results]
#   Pick CHUNK_SIZE so a task runs ~10-30 min, and the %200 throttle to what
#   the cluster is happy with.  A point took ~1 s of one core with the flang
#   image (measured under x86 emulation on a laptop; expect the same or better
#   on the cluster), so CHUNK_SIZE=500 is ~10 min.  gfortran is ~25x slower.
#
# Robustness: `normgrid run` is resumable and crash-tolerant (a point whose
# integral errors out is recorded as nan and the worker restarts past it).
# If tasks are killed (time limit, preemption) or fail, just submit the same
# command again -- finished points are skipped, so only unfinished chunks do
# real work.  Failed points are re-tried with a looser tolerance via
#   python python/cli/normgrid.py run --spec SPEC --start A --stop B --out FILE \
#          --retry-failed --epsrel 1e-6
#
# Environment (all optional):
#   BAND_SIF         container image (default /scratch/$USER/containers/band.sif,
#                    where slurm/pull_container.job puts it; or build one from
#                    Dockerfile_intel or _llvm as in the README).  Set
#                    BAND_NATIVE=1 to skip the container and use the repo's own
#                    python/ and lib/ (i.e. you built the library on the cluster).
#   BAND_PYTHON      python inside the container (default: the image's conda
#                    'band' env; the login-shell conda activation in the
#                    Dockerfile does not happen under apptainer)
#   BAND_CONTAINER_CMD  force apptainer or singularity (default: whichever is
#                    on PATH, preferring apptainer)
#   BAND_PYTHON_FROM_REPO=1  bind this checkout's python/ over the image's
#                    /app/python, so python-side changes need no image
#                    rebuild.  Only safe while the image's Fortran is current.

#SBATCH --job-name=normgrid
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --output=logs/normgrid_%A_%a.out
#SBATCH --error=logs/normgrid_%A_%a.err

set -euo pipefail

: "${SLURM_ARRAY_TASK_ID:?submit as a job array: sbatch --array=0-N-1 slurm/normgrid.sbatch SPEC [SIZE] [DIR] (see the header)}"
SPEC=${1:?usage: slurm/normgrid.sbatch SPEC.json [CHUNK_SIZE] [RESULTS_DIR]}
SIZE=${2:-500}
OUT_DIR=${3:-results}
mkdir -p "$OUT_DIR"

# One thread per task: with an ifx/flang build every worker would otherwise
# start its own OpenMP pool and the tasks would oversubscribe their core.
export OMP_NUM_THREADS=1

if [[ "${BAND_NATIVE:-0}" == "1" ]]; then
    RUN=()          # (expanded as ${RUN[@]+"${RUN[@]}"}: an empty array is an error under set -u on bash < 4.4)
    PY=${BAND_PYTHON:-python}
    NORMGRID=python/cli/normgrid.py
    export LD_LIBRARY_PATH="$PWD/lib:${LD_LIBRARY_PATH:-}"
else
    # apptainer and singularity take the same arguments here; clusters have
    # one or the other (Alderaan ships only singularity-ce), so take whichever
    # is on PATH rather than assuming.  BAND_CONTAINER_CMD forces a choice.
    if [[ -n "${BAND_CONTAINER_CMD:-}" ]]; then
        CONTAINER_CMD=$BAND_CONTAINER_CMD
    elif command -v apptainer >/dev/null 2>&1; then
        CONTAINER_CMD=apptainer
    elif command -v singularity >/dev/null 2>&1; then
        CONTAINER_CMD=singularity
    else
        echo "neither apptainer nor singularity is on PATH on $(hostname);" >&2
        echo "set BAND_CONTAINER_CMD, or BAND_NATIVE=1 to use a local build" >&2
        exit 1
    fi

    BINDS=(--bind "$PWD:$PWD")
    # The image carries both the compiled library (/app/lib) and a copy of
    # python/ taken at build time.  BAND_PYTHON_FROM_REPO=1 binds this
    # checkout's python/ over /app/python so python-side changes do not need
    # an image rebuild -- valid exactly while the Fortran in the image is
    # current, since /app/lib still comes from the image.
    if [[ "${BAND_PYTHON_FROM_REPO:-0}" == "1" ]]; then
        BINDS+=(--bind "$PWD/python:/app/python")
    fi

    SIF=${BAND_SIF:-/scratch/$USER/containers/band.sif}
    [[ -f "$SIF" ]] || { echo "no container at $SIF on $(hostname): set BAND_SIF" >&2; exit 1; }
    RUN=("$CONTAINER_CMD" exec "${BINDS[@]}" --pwd "$PWD" "$SIF")
    PY=${BAND_PYTHON:-/opt/anaconda/envs/band/bin/python}
    NORMGRID=/app/python/cli/normgrid.py
    export APPTAINERENV_LD_LIBRARY_PATH="/app/lib:/opt/anaconda/envs/band/lib"
    export SINGULARITYENV_LD_LIBRARY_PATH=$APPTAINERENV_LD_LIBRARY_PATH   # older singularity
fi

# this task's chunk: line (task id + 1) of the chunk list (a bad spec makes
# the next line fail loudly under `set -e`)
CHUNKS=$(${RUN[@]+"${RUN[@]}"} "$PY" "$NORMGRID" chunks "$SPEC" --size "$SIZE")
read -r START STOP <<< "$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" <<< "$CHUNKS")"
if [[ -z "${START:-}" ]]; then
    echo "array task $SLURM_ARRAY_TASK_ID is past the last chunk of $SPEC; nothing to do"
    exit 0
fi

STEM=$(basename "$SPEC" .json)
OUT="$OUT_DIR/res_${STEM}_${START}.txt"
echo "[$(date +%T)] $STEM points [$START, $STOP) -> $OUT on $(hostname)"

# nonzero exit only means some points failed (recorded as nan in $OUT);
# report it but let the task finish so the result file is kept
${RUN[@]+"${RUN[@]}"} "$PY" "$NORMGRID" run --spec "$SPEC" --start "$START" --stop "$STOP" --out "$OUT" || \
    echo "WARNING: some points in [$START, $STOP) failed; see $OUT (retry with --retry-failed --epsrel 1e-6)"
