#!/bin/bash
# One batch job of the normalization-table build: evaluate grid points
# [START, STOP) of a spec and write result_<TAG>.txt (see python/normgrid.py).
# Runs inside the band container (built from Dockerfile_intel/_llvm and
# converted to band.sif; the repo lives at /app there).  Idempotent and
# crash-tolerant: a point whose integral errors out is recorded as nan and
# the worker restarts past it, so one bad point never loses the whole chunk.
set -u
SPEC=$1; START=$2; STOP=$3; TAG=$4
export LD_LIBRARY_PATH=/app/lib:${LD_LIBRARY_PATH:-}
cd /app
python python/normgrid.py run --spec "$OLDPWD/$SPEC" --start "$START" --stop "$STOP" \
       --out "$OLDPWD/result_${TAG}.txt"
# a nonzero exit here only means some points failed; they are recorded in the
# result file, and the job still succeeds so the file is transferred back
exit 0
