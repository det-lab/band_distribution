"""
Checks the compiled library's reported version (PpqFort_version(), via
_ppqfort_bindings.version()) against fpm.toml's version field.

PpqFort_version()'s constants come from src/version.f90.inc, generated
from fpm.toml by scripts/generate_version_include.py (run before building;
see that script's docstring).  This is the regression guard for the
failure that motivated it: v1.1.0 and v1.1.1 both shipped a compiled
library that reported "1.1.0" because the version used to be a second,
hand-typed copy in Fortran source that nobody remembered to bump.  If the
include is ever stale -- generated once, fpm.toml bumped again without
rerunning the script -- this catches it immediately rather than shipping
it silently.

Needs the compiled Fortran library:
  LD_LIBRARY_PATH=lib python test/python/test_version.py
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
import _ppqfort_bindings
import generate_version_include as genver

failures = []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail and not cond else ''}")
    if not cond:
        failures.append(name)


fpm_version = genver.read_version()
lib_version = _ppqfort_bindings.version()
check("compiled library reports fpm.toml's version", lib_version == fpm_version,
      f"fpm.toml: {fpm_version}, library: {lib_version}")

print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("ALL PASS")
