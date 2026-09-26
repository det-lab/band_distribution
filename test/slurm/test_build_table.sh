#!/bin/bash
# End-to-end test of slurm/build_table.sh with no cluster and no Fortran: a mock
# sbatch runs the array tasks locally, and normgrid's analytic stand-in evaluator
# (NORMGRID_FAKE=smooth: exp(3 u_V + 6 u_q0), a known convergence rate) replaces
# the integrals.  Needs h5py (merge writes HDF5), e.g. inside the container.
#
#   bash test/slurm/test_build_table.sh          (from the repository root)
set -uo pipefail
cd "$(dirname "$0")/../.."
export PATH="$PWD/test/slurm:$PATH" BAND_NATIVE=1 PYTHON=${PYTHON:-python}
export BAND_PYTHON=$PYTHON
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
fail=0
check() { if [[ $2 -eq 0 ]]; then echo "PASS  $1"; else echo "FAIL  $1"; fail=1; fi; }
common="BAND=ER GRID_CHUNK=5000 STUDY_CHUNK=100 HELD_CHUNK=50 HELDOUT=60"

# 1. the smooth stand-in: plan, build, and the table passes validation
NORMGRID_FAKE=smooth WORKDIR=$TMP/w OUT_TABLE=$TMP/t.h5 slurm/build_table.sh ER 2.5 350 0.75 200 > $TMP/ok.log 2>&1
check "full chain succeeds on a smooth problem" $?
grep -q "PASS: worst-case error" $TMP/ok.log; check "validate reports PASS" $?
[[ -f $TMP/t.h5 ]]; check "table written" $?
python - <<PY
import json, sys
r = json.load(open("$TMP/w/tol1e-7/plan.json"))["axes"]
n = {a: v["n"] for a, v in r.items()}
sys.exit(0 if n["q0"] > n["V"] > 3 and all(n[a] == 3 for a in n if a not in ("V", "q0")) else 1)
PY
check "planned nodes: q0 > V > others (=3)" $?

# 2. re-running reuses everything (finished points are skipped)
NORMGRID_FAKE=smooth WORKDIR=$TMP/w OUT_TABLE=$TMP/t.h5 slurm/build_table.sh ER 2.5 350 0.75 200 > $TMP/again.log 2>&1
check "re-run succeeds" $?

# 3. an impossible acceptance level fails the check and says how to retry
NORMGRID_FAKE=smooth WORKDIR=$TMP/w OUT_TABLE=$TMP/t2.h5 ACCEPT=1e-30 slurm/build_table.sh ER 2.5 350 0.75 200 > $TMP/fail.log 2>&1
[[ $? -eq 1 ]]; check "ACCEPT too strict => exit status 1" $?
grep -q "TOL=2e-08 slurm/build_table.sh ER" $TMP/fail.log; check "failure message suggests a tighter TOL" $?

# 4. an axis that never converges stops the chain before the big grid is built
NORMGRID_FAKE=kink WORKDIR=$TMP/k OUT_TABLE=$TMP/t3.h5 slurm/build_table.sh NR 2.5 350 0.75 200 > $TMP/kink.log 2>&1
[[ $? -ne 0 ]]; check "kink: chain stops with an error" $?
grep -q "not resolved by 33 nodes" $TMP/kink.log; check "kink: says which axis and why" $?
[[ ! -e $TMP/k/tol1e-7/spec.json ]]; check "kink: no grid spec was written" $?

# 5. bad usage
slurm/build_table.sh ER 2.5 350 0.75 > /dev/null 2>&1; [[ $? -eq 2 ]]; check "wrong argument count is a usage error" $?
exit $fail
