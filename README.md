This code is useful for dark matter searches where detector output is Ep (total phonon energy) and Eq (total charge energy).  The point of the code is to provide the probability of an (Ep, Eq) pair given a set of detector parameters for both electron recoils (`PpqG`, where the G is for gamma because gammas are the cause of most electron recoils) and neutron recoils (`PpqN`, where the N is for neutron).


# Citing

Citation metadata for the code is in `CITATION.cff` (GitHub's "Cite this repository" button); each release is archived on Zenodo.  The precomputed normalization tables are a separate dataset, [doi:10.5281/zenodo.23048215](https://doi.org/10.5281/zenodo.23048215) (all versions; one version per added fit window), which `python/normtables.py` fetches by name.

# Building and testing with the Fortran Package Manager (`fpm`)

This project uses the Fortran Package Manager (fpm).  You'll need to install that to build this project; please see https://fpm.fortran-lang.org/install/index.html#install for instructions on installing fpm on your system.  Currently (Nov 2025), building from source will install version 0.14 while installing the package via e.g. `conda` will install version 0.12.

The code parallelizes its integration loops with `do concurrent` using locality specifiers, including the Fortran 2023 `reduce` clause, so you need a recent compiler.  The versions below have been verified via the docker containers in this repository:

|Vendor| Version(s)      |  Build/Test Command                                                                                                                        |
|------|-----------------|--------------------------------------------------------------------------------------------------------------------------------------------|
|GNU   | 15.2.0          | `fpm test --compiler gfortran --profile release --flag "-march=native -fopenmp -ftree-parallelize-loops=4 -fcoarray=single -fPIC"`         |
|Intel | 2025.2.1        | `fpm test --compiler ifx --flag "-fpp -O3 -qopenmp -DHAVE_MULTI_IMAGE_SUPPORT=0" --profile release`                                        |
|LLVM  | 22              | `fpm test --compiler flang --profile release --flag "-O3 -fopenmp -fdo-concurrent-to-openmp=host"`                                         |

Notes:
* GNU: gfortran is slow.  It has no option for mapping `do concurrent` onto threads (the auto-parallelizer in `-ftree-parallelize-loops` cannot parallelize these loops because they contain function calls), so the band integrals run **serially** in gfortran builds — measured at 1.0 effective threads vs 17.6 for ifx on the same machine, making per-event likelihood evaluation ~25x slower there (roughly the core count times a ~1.5x per-core gap from the vector math library).  The gfortran build is fine for verifying the physics and the Python interface, but use ifx or flang for production likelihood work (MCMC).
* Intel: the shared library must also be linked against the Intel OpenMP runtime for the Python ctypes interface to work; the Intel container does this by setting `FPM_LDFLAGS="-liomp5"`.  The `-fpp -DHAVE_MULTI_IMAGE_SUPPORT=0` flags are for the Julienne dependency.  `-qopenmp` maps `do concurrent` onto the OpenMP thread pool.
* LLVM: the compiler is invoked as `flang`.  `-fdo-concurrent-to-openmp=host` is what parallelizes the `do concurrent` loops; without it they compile to serial loops.  The LLVM container sets `FPM_LDFLAGS="-fopenmp"` so the shared library links against `libomp`.  Intel and LLVM builds benchmark identically (~6 us per event in vector mode on 18 emulated cores).

## Building the shared library for the python interface
The python wrappers load `lib/libband_distribution.so`, which `fpm install` builds and places under the repository root.  Use the same flags as the test commands above; for Intel and LLVM, `FPM_LDFLAGS` must also be set so the *shared library* links its OpenMP runtime — without it the library builds but fails to load from python with undefined `__kmpc_*` symbols:

```
# Intel
FPM_LDFLAGS="-liomp5" fpm install --prefix=. --compiler ifx --profile release --flag "-fpp -O3 -qopenmp -DHAVE_MULTI_IMAGE_SUPPORT=0"

# LLVM
FPM_LDFLAGS="-fopenmp" fpm install --prefix=. --compiler flang --profile release --flag "-O3 -fopenmp -fdo-concurrent-to-openmp=host"

# GNU (verification only — see the notes above)
fpm install --prefix=. --compiler gfortran --profile release --flag "-march=native -fopenmp -ftree-parallelize-loops=4 -fcoarray=single -fPIC"
```

# Testing the python calls
This code builds a library that may be called within python (this is the original intent of the code).  Build and install the shared library first (see "Building the shared library for the python interface" above).  The python test scripts live in `test/python/` and should be run from the repository root.  To test the python calls, run

```
LD_LIBRARY_PATH=lib python test/python/test_PpqFort.py
```

or if you just want to test the vectorized functions `PpqN_vector` and `PpqG_vector`

```
LD_LIBRARY_PATH=lib python test/python/test_PpqFort_vectorFuncs.py
```

(`LD_LIBRARY_PATH=lib` is needed when running outside the docker containers, which set it in their environment.)

See `test/python/README.md` for a map of the test scripts and their supporting modules, including where the PDF-over-bin integration lives.

# PDF self-consistency tests
These verify that data sampled from the PDFs produces Pearson chi-square values that follow the theoretical chi2(n_bins − 1) distribution.  Reference plots are committed in `figures/`.

```
python test/python/verify_sample_from_pdf.py                       # sampler moment checks (~2 s)
LD_LIBRARY_PATH=lib python test/python/test_chisquare_ppqn.py      # Fortran PpqN PDF (~2 min first run)
```

`test_chisquare_ppqn.py` takes the same two optional positional arguments as the simulator tests below: `n_throws` (default 100,000) and `n_bins` (default 64).  For a quick smoke test with fewer throws, run

```
LD_LIBRARY_PATH=lib python test/python/test_chisquare_ppqn.py 2000 64
```

It caches its PDF grid evaluation in `ppqn_vertex_grid.npz` (not committed) and reuses it on later runs.

The `test_chisquare_ppqn.py` timing assumes an ifx or flang build; gfortran builds run the band integrals serially (see the compiler notes above) and take correspondingly longer.

# Physics-simulator validation tests
The tests above only check that the PDFs are *self-consistent* (samples drawn from a PDF match that same PDF).  The two tests below are the stronger check: they generate events from an independent physics simulator (`test/python/generate_events.py`, which draws Er from the recoil spectrum, N from a truncated normal, and applies detector resolution) and compare the binned counts against the Fortran PDFs.  A pass means the Fortran `PpqN` / `PpqG` implementations correctly describe the physics.

```
LD_LIBRARY_PATH=lib python test/python/test_chisquare_nr_simulator.py [n_throws] [n_bins]   # NR band vs PpqN
LD_LIBRARY_PATH=lib python test/python/test_chisquare_er_simulator.py [n_throws] [n_bins]   # ER band vs PpqG
```

Both write their chi-square histograms to `figures/chisquare_nr_simulator.png` and `figures/chisquare_er_simulator.png`.  `n_throws` defaults to 1000 and `n_bins` to 400.  The one-time integration of the PDF over the bins dominates the wall time, so reducing `n_bins` is the way to get a fast smoke test:

```
# smoke test: 100 throws, 64 bins
LD_LIBRARY_PATH=lib python test/python/test_chisquare_nr_simulator.py 100 64   # ~1 min
LD_LIBRARY_PATH=lib python test/python/test_chisquare_er_simulator.py 100 64   # ~1 min

# full validation: 10,000 throws, 400 bins
LD_LIBRARY_PATH=lib python test/python/test_chisquare_nr_simulator.py 10000    # ~4 min
LD_LIBRARY_PATH=lib python test/python/test_chisquare_er_simulator.py 10000    # ~5 min
```

(Timings measured with the ifx build, 18 workers, under x86 emulation on an Apple Silicon Mac; native x86 hardware should be faster.  gfortran builds run the band integrals serially and will be dramatically slower here — use ifx or flang.)

To run inside the Intel docker container, mount the repository's `figures/` directory so the plot survives the container:

```
docker run --rm -v $(pwd)/figures:/app/figures band_distribution_intel \
    bash --login -c "conda activate band && python /app/test/python/test_chisquare_nr_simulator.py 100 64"
```

# Performance notes
`PpqN` / `PpqG` integrate over a window placed around the located peak(s) of the Er integrand rather than sampling the full physical range, evaluating at roughly 6 microseconds per (Ep, Eq) point in vector mode (18 threads under x86 emulation; measured via `PpqN_vector` over a representative grid).  This number assumes an ifx or flang build — gfortran builds run the band integrals serially and are ~25x slower (see the compiler notes at the top).  For likelihood loops (e.g. MCMC):

* Call the vectorized entry points (`PpqN_vector` / `PpqG_vector`) with all events in one call — the parallelism lives there, and per-event scalar calls pay OpenMP fork/join overhead instead.
* **Shuffle the event array once at load time if it is ordered.**  The vector loops split the events into one contiguous chunk per thread (static scheduling), and the loop only finishes when the slowest chunk does.  Per-event cost varies several-fold across the (Ep, Eq) plane — deep-tail events short-circuit in ~2 us while on-band events cost ~10-15 us — so an energy-ordered array hands some threads chunks of expensive events while others idle at the barrier.  Shuffling gives every chunk a similar cost mix and measured 8-15% faster than energy-ordered input.  The result is identical either way, and if your events are already in effectively random order this changes nothing.

# Normalizing a likelihood fit to a region: `PpqPDF`

A likelihood fit needs the *un-normalized* PDF at each data point and the PDF's integral over the fit region (to normalize it) — both re-evaluated at every step as the fit/MCMC explores parameter space.  Computing that normalization with `scipy.integrate.quad` is several seconds per call (it evaluates the PDF one point at a time, at the ~3-25 ms/point *scalar* cost — see `python/ppqfort_pdf.py`'s docstring); `python/ppq_pdf.py`'s `PpqPDF` instead uses a doubling-verified nested Gauss-Legendre quadrature evaluated in one batched, thread-parallel call (`PpqN_region`/`PpqG_region` in Fortran — the ridge location/width is already known exactly, same physics as `test/python/band_breakpoints.py`'s ridge/width derivation, so it needs far fewer points than a naive grid for the same accuracy), landing well under a second per call even for a wide region under `ifx`/`flang`.

`PpqPDF` bundles the things that *don't* change across a fit/MCMC run — the region, the observed dataset, and the normalization quadrature's convergence tolerance — at construction, so every subsequent call only needs the physics parameters that the fit is actually varying:

```python
import sys
sys.path.insert(0, "python")  # or wherever your checkout's python/ dir lives
import numpy as np
from ppq_pdf import PpqPDF

fit = PpqPDF(ep_min, ep_max, eq_min, eq_max, ep_data, eq_data,
             norm_epsrel=1e-4, norm_epsabs=1e-10)
# norm_epsrel/norm_epsabs set how tightly two successive doubled
# quadrature orders must agree before the normalization integral is
# trusted (see PpqN_region's doc comment in src/PpqFort_m.f90) -- NOT
# related to len(ep_data).  1e-4/1e-10 agreed with scipy.integrate.quad
# to ~1e-12 relative across the regions tested in
# test_region_integral.py, including a 247 keV-wide region.

def loglike(k, Z, F0, eps, V, p0, p10, q0, q10):
    vals = fit.ppqn_values(k=k, Z=Z, F0=F0, eps=eps, V=V, p0=p0, p10=p10, q0=q0, q10=q10)
    norm = fit.ppqn_integral(k=k, Z=Z, F0=F0, eps=eps, V=V, p0=p0, p10=p10, q0=q0, q10=q10)
    return np.sum(np.log(vals)) - len(fit.ep_data) * np.log(norm)
    # or fit.ppqn_normalized_values(...) if you want vals/norm directly
```

Same shape for the ER band (`ppqg_integral`/`ppqg_values`/`ppqg_normalized_values`, no `k`/`Z` — `Y=1` there). Build `fit` once, outside the fit loop; call its methods once per step, inside.

# Precomputed normalization tables for MCMC: `python/normgrid.py`

Even at ~2 s, computing the normalization integral at every MCMC step is out of the question.  But the region is fixed for a whole run and the integral is a very smooth function of the physics parameters the MCMC varies, so `normgrid.py` evaluates it once on a small tensor grid (one independent integral per grid point — a good fit for the OSG) and interpolates it at tens of microseconds per step.  Separate tables for the NR band (`k, F0, V, p0, p10, q0, q10`) and the ER band (same, no `k`); `Z` and `eps` are fixed inside a table and checked on every call.

A few design points worth knowing:

* **Axes.** `k`, `F0` (used *linearly* — the integral is smooth in F0 but not in log F0, so 3 nodes give ~3e-9 where log F0 needs 9+), `V`, `p0`, `dp = p10 - p0`, `q0`, `dq = q10 - q0`.  The resolution model `sigp² = p0² + (p10² - p0²)(Ep/c)²` is only defined for `p10 >= p0` (likewise `q10 >= q0`), so a plain `(p0, p10)` or `(q0, q10)` box would contain unphysical corners (the `q0` and `q10` ranges overlap); `dp >= 0` and `dq >= 0` keep the box rectangular.  Query the table with the physical `p10`/`q10`; it converts.  `PpqN_region`/`PpqG_region` now error out in ~0.3 s on such inputs (and on `F0, eps, p0, q0, k, Z <= 0`, an empty region, or NaN) instead of grinding for minutes.
* **Interpolant.** Tensor-product Chebyshev series through Chebyshev–Lobatto nodes, built and evaluated with `numpy.polynomial.chebyshev` (`chebfit`/`chebval`; exact at the nodes, ~40 µs per lookup natively — about 0.1% of a 20,000-event likelihood evaluation).  It **never extrapolates** — a query outside the table's box raises `OutOfBoxError`, so keep the MCMC prior inside the box (see [Fits and MCMC with the tables](#fits-and-mcmc-with-the-tables); default box: `k` 0.13–0.22, `F0` 1e-5–1 (ER: 0.1–0.35, bracketing the effective low-field electron-recoil Fano factor of ~0.2–0.3), `V` 2.7–3.3, `p0` 0.0128–0.1156 and `q0` 0.0474–0.4 (the ±4σ range of Gaussian priors centred on 0.0642 and 0.2372 with a 20% 1σ width, `q0` capped at the maximum `q10`), `dp = p10-p0` 0.18–0.59 and `dq = q10-q0` 0–0.353 (from `p10` 0.3–0.6, `q10` 0.2–0.4); for a different one, rebuild with `BOX=`, e.g. `BOX="V=2.5:4" sbatch slurm/build_table.job ...`, or change the defaults `DEFAULT_BOX` / `BAND_BOX_OVERRIDES`).
* **Node counts** (`RECOMMENDED_NODES`, from per-axis studies against directly computed held-out points; region-dependent): NR `k=8 F0=3 V=4 p0=4 dp=3 q0=16 dq=4` (73,728 points); ER `F0=3 V=9 p0=4 dp=6 q0=11 dq=9` (64,152 points).  The counts depend on the region: these are for the analysis ROI Ep 2.5–350 / Eq 0.75–200 (the earlier 2–200 / 4–100 region needed far fewer, e.g. 5 nodes in `q0`).  `q0` is the hard axis for NR — its error falls only ~4× per two nodes — so check it with `validate`.  At roughly 1–3 s per point on one core that is tens of core-hours.  These target ~1e-7 worst-case error per axis, i.e. ~0.02 in the log-likelihood at 20,000 events (the error is `N_events × δN/N`).  Always confirm with `validate` — the per-axis studies cannot see cross terms.

```
# describe the grid (region and epsrel are required; nodes default to RECOMMENDED_NODES
# and the box to the one above -- a wrong region makes a table that looks fine and isn't)
python python/normgrid.py make-spec --band NR --region 2 200 4 100 --epsrel 1e-7 --out spec_NR.json
python python/normgrid.py make-random-spec --band NR --n 200 --region 2 200 4 100 --epsrel 1e-7 --out held_NR.json

# run it: on one machine, or as batch jobs (osg/normgrid.sub + osg/normgrid_job.sh are an HTCondor template)
python python/normgrid.py chunks spec_NR.json --size 100        # "start stop" ranges, one per job
python python/normgrid.py run --spec spec_NR.json --start 0 --stop 100 --out res_0.txt
python python/normgrid.py run --spec held_NR.json --out res_held.txt

# combine into one HDF5 file and check it against the held-out points
python python/normgrid.py merge --spec spec_NR.json --results 'res_*.txt' --out norm_NR.h5
python python/normgrid.py validate --table norm_NR.h5 --heldout-spec held_NR.json --results res_held.txt
```

**On a Slurm cluster** use the **ifx (or flang) container, not gfortran** — gfortran runs the band integrals ~25x slower — via `slurm/normgrid.sbatch`, a job-array template (one single-core task per chunk of grid points; `OMP_NUM_THREADS=1` so an ifx build doesn't oversubscribe cores; a header comment gives the exact `sbatch` and `merge`/`validate` commands, and resubmitting the same array only redoes unfinished chunks).  It runs the container `$BAND_SIF` (default `/scratch/$USER/containers/band.sif`, as pulled by `slurm/pull_container.job`), a `band.sif` built from `Dockerfile_intel` or `Dockerfile_llvm` (see "Build the singularity/apptainer container" below), or set `BAND_NATIVE=1` to use a library you built on the cluster.  `osg/` holds a similar HTCondor template.

**One command for a whole table.**  How many nodes an axis needs depends on the fit region (the same box needed 5 nodes in `q0` for Eq 4–100 and about 16 for Eq 0.75–200), so `slurm/build_table.sh` measures instead of assuming.  For a region it (1) evaluates 33 Chebyshev–Lobatto nodes along each axis at three baseline points (`python/normplan.py points`, about 700 integrals), (2) reads the Chebyshev coefficients of each axis to find the fewest nodes whose truncation error is below `TOL` (`normplan.py analyze`; it stops with an error if an axis is not resolved by 33 nodes, rather than guessing), (3) evaluates the resulting grid and a set of held-out points, and (4) merges and validates, exiting with status 1 if the worst held-out error exceeds `ACCEPT`:

```
sbatch slurm/build_table.job NR 2.5 350 0.75 200              # -> tables/norm_NR_ep2.5-350_eq0.75-200.h5
TOL=2.5e-8 sbatch slurm/build_table.job NR 2.5 350 0.75 200   # retry with more nodes; earlier results are reused
```

Each Slurm stage is `sbatch --wait slurm/normgrid.sbatch ...`, so `slurm/build_table.sh` runs until the whole chain is done; `slurm/build_table.job` runs it as a single-core driver job that waits in the queue like any other (the environment at submission, e.g. `TOL`, is passed on).  Every array is checked with `sacct` when it ends and a failed task stops the chain (`sbatch --wait` exits 0 on Alderaan even when tasks fail).  The container is `BAND_SIF`, by default `/scratch/$USER/containers/band.sif` where `slurm/pull_container.job` puts it.  Options (`TOL`, `EPSREL`, `ACCEPT`, chunk sizes, `ACCOUNT`/`PARTITION`/`TIME`) are documented in the script header.  The one-axis-at-a-time study cannot see cross terms between axes, which is why the held-out check is the gate.  `test/slurm/test_build_table.sh` runs the whole chain locally against a mock `sbatch` and an analytic stand-in for the integrals.

`run` is resumable and crash-tolerant: Fortran `error stop` (e.g. the quadrature not certifying `epsrel=1e-7`) kills the process, so a supervisor records that point as `nan` and restarts past it; re-run failures with a looser tolerance via `run --retry-failed --epsrel 1e-6`.  `merge` refuses to build a table with missing points.  The HDF5 file records the region, fixed parameters, library version and git commit it was built with — rebuild if any of those change.  (`h5py` is in `environment.yaml`.)  Then hand the tables to `PpqPDF`; nothing else in the fit changes:

```python
fit = PpqPDF(ep_min, ep_max, eq_min, eq_max, ep_data, eq_data,
             ppqn_table="norm_NR.h5", ppqg_table="norm_ER.h5")   # region must match the tables
# fit.ppqn_integral(...) / fit.ppqg_integral(...) now interpolate instead of integrating
```

**Published tables** are listed in `python/table_registry.json` and can be passed by name instead of a path: `python/normtables.py` downloads them from [Zenodo](https://doi.org/10.5281/zenodo.23048215) on first use, checks their SHA-256, and caches them (`$BAND_TABLES_DIR`, else `~/.cache/band_distribution`; the containers ship with every registered table in `/app/tables`).  `python python/normtables.py list` shows what is available.

# Fits and MCMC with the tables

A table covers a fixed box of parameters and never extrapolates, so the rule for a fit or an MCMC is the same: **keep every parameter inside the tables' box, with `p10 >= p0` and `q10 >= q0`** (the PDF is undefined otherwise).  The library checks this and says what to change; the fit or MCMC code has to enforce it.

* `PpqPDF(..., prior_bounds=BOUNDS)` checks hard bounds against every table given, at construction.  If they reach outside a table it raises `PriorNotCoveredError`, listing each short axis and the `BOX=` rebuild command that would cover them.
* `pdf.table_bounds()` gives, per band, the widest bounds each table covers: a ready-made set of hard bounds.  Take bounds from it, or round inward: the box edges are not round numbers (`p0` starts at 0.0128438), and bounds rounded outward fail the check.
* `pdf.check_points(points)` checks a set of points, e.g. an MCMC's starting walkers.
* At run time, a query outside a table raises `OutOfBoxError` naming the axis, the value, the box and the table file.  After the setup checks pass, that means a bug or a NaN.

Bounds are physical parameters, `{name: (lo, hi)}`.  Parameters the two bands share go at the top level; each band's own (NR's `k`, and the Fano factor `F0`, which differs between the bands) go under `"NR"` and `"ER"`.  `Z` and `eps` are fixed inside a table; pass them as their fixed values or leave them out.

## MCMC

```python
BOUNDS = {"V": (2.7, 3.3), "p0": (0.0129, 0.1155), "p10": (0.3, 0.6),
          "q0": (0.0475, 0.4), "q10": (0.2, 0.4),
          "NR": {"k": (0.13, 0.22), "F0": (1e-5, 1.0)},
          "ER": {"F0": (0.1, 0.35)}}

pdf = PpqPDF(2.5, 350, 0.75, 200, ep_data, eq_data,
             ppqn_table="NR_ep2.5-350_eq0.75-200", ppqg_table="ER_ep2.5-350_eq0.75-200",
             prior_bounds=BOUNDS)                  # 1. fails now, not hours into the run

def log_prior(theta):                             # 2. -inf outside the hard bounds and the ordering
    p = unpack(theta)
    if not within(p, BOUNDS) or p["p10"] < p["p0"] or p["q10"] < p["q0"]:
        return -np.inf
    return soft_terms(p)                          # e.g. Gaussians on p0, q0 -- truncated by BOUNDS

def log_prob(theta):                              # 3. the prior first: the table never sees
    lp = log_prior(theta)                         #    a point outside it
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, pdf)

pdf.check_points(as_params(start))               # 4. starting walkers inside, before sampling
                                                  #    ({param: array}, laid out like BOUNDS)
```

A Gaussian prior (on `p0` or `q0`, say) must be truncated by the hard bounds: `prior_bounds` rejects a parameter without finite bounds, since no table can cover it.  Walkers started in a small ball around a best fit can still land outside near an edge, which is what step 4 catches.

## Fits (maximum likelihood / MAP)

* Give the optimizer bounds inside the tables (e.g. `bounds=` for `scipy.optimize.minimize` with L-BFGS-B, Powell or trust-constr), and check them once with `prior_bounds` or `table_bounds()`.
* Box bounds cannot express `p10 >= p0`.  Either fit `dp = p10 - p0` and `dq = q10 - q0` directly, bounded by the tables' own `dp`/`dq` axes (`pdf.ppqn_table.box["dp"]`, 0.18–0.59 by default; `dq` 0–0.353), and convert back when calling the PDF, or add the inequalities as constraints (trust-constr, SLSQP).
* A fit that ends on a bound is being limited by the box, not the data: widen it with `BOX=` and rebuild the tables.

# Build the singularity/apptainer container for HPC submissions
There are multiple Dockerfiles, each building the code with a compiler from a different vendor (GNU, Intel, and LLVM).  

|Vendor| Dockerfile name     | Notes |
|------|---------------------|-------|
|GNU   | Dockerfile_gfortran | Slow: gfortran runs the band integrals serially, ~25x slower than ifx/flang (see the compiler notes above) |
|Intel | Dockerfile_intel    | Recommended for work that needs speed |
|LLVM  | Dockerfile_llvm     | Same performance as the Intel build |

Choose which compiler you want, determine the name of the dockerfile, and then issue the following command:

```
docker build -f {dockerfile name} -t band .
```

By default this builds from your local checkout, including any uncommitted changes.  To instead build a specific released version — e.g. so someone else can reproduce your exact results without needing their own clone lined up to the right commit — pass `GIT_REF` as a build argument.  It clones the repository fresh from GitHub and checks out that tag or commit hash in place of the local files:

```
docker build -f {dockerfile name} -t band --build-arg GIT_REF=v0.9.0 .
```

`GIT_REF` accepts any tag or commit hash (run `git tag -l` to see available versions) and works identically on all five Dockerfiles (`Dockerfile_gfortran`, `Dockerfile_intel`, `Dockerfile_llvm`, `Dockerfile_jupyter`, `Dockerfile_tau_intel`).  `v0.9.0` is the last release before the Lindhard `(k, Z)` yield model's breaking interface change.

If you need to troubleshoot the docker build, you can shell into this container with the command

```
docker run -it band
```

Now you have a docker container that contains the fortran binary, but this is not usable on HPC systems.  Run this command to create `band.sif`, an image file that can be used on HPC systems.  The command can be issued in any location (it is not directory dependent).

```
apptainer build band.sif docker-daemon://band:latest
```

# Build the docker container for running Jupyter and interacting with notebooks

```
docker build --rm -f Dockerfile_jupyter -t band_jupyter .
```

## Run the jupyter container
You can issue this command from any directory.  Note the absolute path names for mounting the volume.  This enables your work to persist!  You will need to replace `/mnt/c/Users/canto/Repositories/nrFanoII` with the path to your repository directory.  You should leave `home/jovyan/work/nrFano` the same.  Note that this command refers to the nrFanoII repository, which uses this (band_distribution) repository.

```
docker run -it --rm -p 8888:8888 -v /mnt/c/Users/canto/Repositories/nrFanoII:/home/jovyan/work/nrFano band_jupyter:latest
```

In notebooks, select the **Python (band)** kernel — it runs in the `band` conda environment (built from `environment.yaml`, the same environment used by the other containers), which has the compiled library's runtime dependencies and all the python packages.

# Use the docker container for local development
For local development, you most likely want the files available to you in a way that persists once you close the container.  In this case you need to supply arguments to `docker run` that mount the top-level directory:

```
docker run -it --mount type=bind,src=.,dst=/app --entrypoint=/bin/bash band
```

# Profiling with TAU
TAU is built into `Dockerfile_tau_intel`. This image uses the ifx compiler because that comipler is the easiest to install outside a Docker container.  Build that image first:

```
docker build -f Dockerfile_tau_intel -t tau_intel .
```

Once the image is built, you can run the container and shell into it with. The repository is mounted at `/repo` so that any changes to files persist on the host. The built library remains at `/app`.

```
# for fish shell
docker run -it -v /tmp/.X11-unix:/tmp/.X11-unix -v "$XAUTH/.Xauthority:/root/.Xauthority" -v $PWD:/repo tau_intel /bin/bash

# for bash shell
docker run -it -v /tmp/.X11-unix:/tmp/.X11-unix -v "$XAUTH/.Xauthority:/root/.Xauthority" -v ($pwd):/repo tau_intel /bin/bash
```

Inside the docker prompt, you'll need to set the DISPLAY environment variable.  After this, tools that require a GUI window like `paraprof` will work.

```
export DISPLAY=:0.0
```

Set the TAU environment variables:

```
export TAU_MAKEFILE=/packages/tau2/x86_64/lib/Makefile.tau-pthread
export TAU_OPTIONS="-optCompInst -optVerbose -optNoRevert"
export TAU_THROTTLE=0
```

Compile the profiling driver using `tau_f90.sh`, which instruments every Fortran source file.
The module and submodule must be compiled before the driver (the `-c` step generates the `.mod` files needed downstream):

```
cd /tmp

tau_f90.sh -O3 -g -qopenmp -fpp -I/app/include -c /app/src/PpqFort_m.f90
tau_f90.sh -O3 -g -qopenmp -fpp -I/app/include -c /app/src/PpqFort_s.f90

tau_f90.sh -O3 -g -qopenmp -fpp \
    -I/app/include \
    PpqFort_m.o PpqFort_s.o /repo/app/profile_driver.f90 \
    -L/app/lib -lband_distribution \
    -Wl,-rpath,/app/lib \
    -o /repo/profile_driver
```

Run the instrumented binary directly (no `tau_exec` needed with compiler instrumentation):

```
/repo/profile_driver
```

Running the code will produce `profile.*` files in the current directory (`/tmp` if you followed the steps above), which you can investigate using `pprof` (terminal summary) or `paraprof` (GUI view).

## Re-profiling after a source change

If you edit source files in `/repo` and want to re-profile, rebuild the library from `/repo` first, then recompile the profile driver against the new library.

**Step 1: Rebuild the library**
```
cd /repo
fpm install --compiler ifx --flag "-fpp -O3 -qopenmp -DHAVE_MULTI_IMAGE_SUPPORT=0" --profile release --prefix /tmp/band_new
```

**Step 2: Add the dependency libraries to `LD_LIBRARY_PATH`**

`fpm install` copies the main library but not its dependencies (`libassert.so`, `libjulienne.so`). Point the linker at the fpm build directory:
```
export LD_LIBRARY_PATH=$(find /repo/build -name "libassert.so" -printf "%h\n" | head -1):$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$(find /repo/build -name "libjulienne.so" -printf "%h\n" | head -1):$LD_LIBRARY_PATH
```

**Step 3: Recompile the profile driver with TAU instrumentation**
```
cd /tmp

tau_f90.sh -O3 -g -qopenmp -fpp -I/tmp/band_new/include -c /repo/src/PpqFort_m.f90
tau_f90.sh -O3 -g -qopenmp -fpp -I/tmp/band_new/include -c /repo/src/PpqFort_s.f90

tau_f90.sh -O3 -g -qopenmp -fpp \
    -I/tmp/band_new/include \
    PpqFort_m.o PpqFort_s.o /repo/app/profile_driver.f90 \
    -L/tmp/band_new/lib -lband_distribution \
    -Wl,-rpath,/tmp/band_new/lib \
    -o /repo/profile_driver
```

**Step 4: Run and inspect**
```
/repo/profile_driver
pprof
```

# Documentation
With [ford](https://github.com/Fortran-FOSS-Programmers/ford) installed, run `ford ford.md`.
Then open `doc/html/index.html` in a web browser to see the band_distribution documentation.
