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
export MOCK_SLURM_DIR=$TMP
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
grep -q "TOL=2e-08 sbatch slurm/build_table.job ER" $TMP/fail.log; check "failure message suggests a tighter TOL" $?

# 4. an axis that never converges stops the chain before the big grid is built
NORMGRID_FAKE=kink WORKDIR=$TMP/k OUT_TABLE=$TMP/t3.h5 slurm/build_table.sh NR 2.5 350 0.75 200 > $TMP/kink.log 2>&1
[[ $? -ne 0 ]]; check "kink: chain stops with an error" $?
grep -q "not resolved by 33 nodes" $TMP/kink.log; check "kink: says which axis and why" $?
[[ ! -e $TMP/k/tol1e-7/spec.json ]]; check "kink: no grid spec was written" $?

# 5. a failed array task stops the chain, although (as on Alderaan) sbatch --wait exits 0
NORMGRID_FAKE=smooth BAND_PYTHON=false WORKDIR=$TMP/f OUT_TABLE=$TMP/t4.h5 slurm/build_table.sh ER 2.5 350 0.75 200 > $TMP/taskfail.log 2>&1
[[ $? -ne 0 ]]; check "failed task: chain stops with an error" $?
grep -q "not every one of its .* tasks COMPLETED" $TMP/taskfail.log; check "failed task: says which array job" $?
[[ ! -e $TMP/f/tol1e-7/plan.json ]]; check "failed task: stopped before the plan" $?

# 6. a custom box: tagged with its hash, and used by every spec
NORMGRID_FAKE=smooth BOX="V=2.6:3.4 dq=0:0.4" WORKDIR=$TMP/b OUT_TABLE=$TMP/b.h5 slurm/build_table.sh ER 2.5 350 0.75 200 > $TMP/box.log 2>&1
check "custom box: chain succeeds" $?
grep -q "(ER_ep2.5-350_eq0.75-200_box[0-9a-f]\{8\})" $TMP/box.log; check "custom box: tag carries the box hash" $?
python - <<PY
import json, sys
boxes = [json.load(open(f"$TMP/b/{f}"))["box"] for f in ("study.json", "held.json", "tol1e-7/spec.json")]
sys.exit(0 if all(b["V"] == [2.6, 3.4] and b["dq"] == [0.0, 0.4] and b["F0"] == [0.1, 0.35] for b in boxes) else 1)
PY
check "custom box: study, held-out and grid specs all use it" $?
jobs_before=$(cat $MOCK_SLURM_DIR/last_job)
BOX="k=0.1:0.2" WORKDIR=$TMP/bad slurm/build_table.sh ER 2.5 350 0.75 200 > $TMP/badbox.log 2>&1
[[ $? -ne 0 ]] && grep -q "not an axis of the ER table" $TMP/badbox.log && [[ $(cat $MOCK_SLURM_DIR/last_job) == "$jobs_before" ]]
check "bad box: fails before anything is submitted" $?

# 7. bad usage
slurm/build_table.sh ER 2.5 350 0.75 > /dev/null 2>&1; [[ $? -eq 2 ]]; check "wrong argument count is a usage error" $?
exit $fail
