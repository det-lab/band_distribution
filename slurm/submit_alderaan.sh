#!/bin/bash
# Submit the normgrid tables for the Ep 2.5-350 / Eq 0.75-200 window on
# math-alderaan.  Wraps slurm/normgrid.sbatch with this cluster's
# account/partition/time (sbatch command-line flags override the
# template's #SBATCH lines, so the template itself stays generic).
#
#   ./slurm/submit_alderaan.sh            # both bands, grid + held-out
#   ./slurm/submit_alderaan.sh NR         # one band only
#
# Re-running is safe and cheap: `normgrid run` skips points that are
# already in the result files, so a resubmission only redoes unfinished
# chunks (use it after a time-limit kill or preemption).
#
# Requires $BAND_SIF (default /scratch/$USER/containers/band.sif, as
# produced by slurm/pull_container.job) built from *this* branch
# (region_normalization) -- the container runs its own
# /app/python/normgrid.py and needs PpqN_region/PpqG_region in /app/lib.

set -euo pipefail
cd "$(dirname "$0")/.."

ACCOUNT=${ACCOUNT:-dark-matter-salting}
PARTITION=${PARTITION:-math-alderaan-short}
TIME=${TIME:-08:00:00}
GRID_CHUNK=${GRID_CHUNK:-500}   # points per array task (~1 s/point under flang)
HELD_CHUNK=${HELD_CHUNK:-100}
THROTTLE=${THROTTLE:-200}        # max concurrent array tasks

: "${BAND_SIF:=/scratch/$USER/containers/band.sif}"
[[ -f "$BAND_SIF" ]] || { echo "no container at $BAND_SIF (set BAND_SIF)" >&2; exit 1; }
export BAND_SIF


mkdir -p logs results tables

submit() {   # submit SPEC CHUNK_SIZE [extra sbatch args...]
    local spec=$1 size=$2; shift 2
    local n
    n=$(python3 python/normgrid.py chunks "$spec" --size "$size" | wc -l)
    echo -n "$(basename "$spec" .json): $n tasks -> "
    sbatch --account="$ACCOUNT" --partition="$PARTITION" --time="$TIME" \
           --array=0-$((n - 1))%"$THROTTLE" --job-name="ng_$(basename "$spec" .json)" \
           "$@" slurm/normgrid.sbatch "$spec" "$size" results
}

BANDS=("$@")
[[ ${#BANDS[@]} -eq 0 ]] && BANDS=(NR ER)

for band in "${BANDS[@]}"; do
    submit "specs/spec_${band}.json" "$GRID_CHUNK"
    submit "specs/held_${band}.json" "$HELD_CHUNK"
done

cat <<EOF

When the arrays finish (squeue -u \$USER), build and check the tables:

  for b in NR ER; do
    python3 python/normgrid.py merge --spec specs/spec_\$b.json \\
        --results "results/res_spec_\${b}_*.txt" \\
        --out tables/norm_\${b}_ep2.5-350_eq0.75-200.h5
    python3 python/normgrid.py validate --table tables/norm_\${b}_ep2.5-350_eq0.75-200.h5 \\
        --heldout-spec specs/held_\$b.json --results "results/res_held_\${b}_*.txt"
  done

merge refuses to write a table with missing points; re-run those with
  normgrid run --spec SPEC --start A --stop B --out FILE --retry-failed --epsrel 1e-6
EOF
