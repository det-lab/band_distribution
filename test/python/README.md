# Python tests

Everything in this directory validates the band-distribution PDFs; the code
that *computes* the PDFs lives in `python/` at the repository root.  If
you're looking for how to *use* the library (a fit or MCMC), that's
`PpqPDF` in `python/ppq_pdf.py` — these tests validate the primitives it's
built from (`python/internal/_ppqfort_bindings.py`, private, and
`pq_dist_v10.py`, the pure-Python reference implementation), not a second way to use it;
each test that needs the raw, unnormalized PDF says why in a comment at its
import.  Run all scripts from the repository root; see the main README for
expected timings and for how to build `lib/libband_distribution.so` first.

## Test scripts

| Script | What it checks |
|--------|----------------|
| `test_PpqFort.py` | Fortran functions against the python reference implementation, one value at a time (machine-precision parity). |
| `test_PpqFort_vectorFuncs.py` | The vectorized `PpqN_vector` / `PpqG_vector` entry points. |
| `verify_sample_from_pdf.py` | Moments of the grid sampler used by the self-consistency tests. |
| `test_chisquare_ppqn.py` | Self-consistency: events sampled *from* the Fortran `PpqN` follow `PpqN`. |
| `test_chisquare_nr_simulator.py` | Physics: events from the independent NR simulator follow the Fortran `PpqN`. |
| `test_chisquare_er_simulator.py` | Physics: events from the independent ER simulator follow the Fortran `PpqG`. |
| `test_region_integral.py` | The region-normalization integral (`ppqn_region`/`ppqg_region`) against nested `scipy.integrate.quad`, timing, the non-convergence safety net, and the fast error-out on invalid parameters (p10 < p0 etc.). |
| `test_normgrid.py` | The precomputed-table pipeline (`python/cli/normgrid.py`): interpolant exactness, no-extrapolation, crash-tolerant worker, HDF5 round trip. No Fortran needed. |
| `test_normplan.py` | The node-count planner (`python/cli/normplan.py`): study layout, the coefficient-tail criterion on functions of known smoothness, the study → analyze path with analytic stand-in evaluators, and that a rough axis stops the plan.  `test/slurm/test_build_table.sh` runs `slurm/build_table.sh` end to end against a mock `sbatch` (needs h5py). |
| `test_normtables.py` | Published tables fetched by name (`python/cli/normtables.py`): register/fetch round trip, hash verification, cache reuse, a bad-hash cached file is not trusted. Needs `pooch`. |
| `test_ppq_pdf_tables.py` | `PpqPDF`'s MCMC setup checks against tables: `prior_bounds` (shared and band-specific keys), `check_points`, `table_bounds`, and an unregistered table name. |
| `test_version.py` | The compiled library's reported version (`PpqFort_version()`) matches `fpm.toml`'s `version` field — the regression guard for `scripts/generate_version_include.py` (see the main README's fpm section). |

The two simulator tests are the strongest statement: the event generator
(`generate_events.py`) knows nothing about the PDF's integrals — it draws
Er from the recoil spectrum, N from a truncated normal, and applies the
detector response — so agreement means the PDFs describe the physics.

## Supporting modules

| Module | Role |
|--------|------|
| `chisquare_harness.py` | PDF-agnostic machinery: equibin binning, expected counts by per-bin quadrature, batched chi-square throws, and a `debug_bin` helper for inspecting one bin's integration in isolation.  Every band PDF here is a narrow ridge, so the per-bin quadrature requires an `inner_points_func` (breakpoints). |
| `band_breakpoints.py` | Band physics for the per-bin quadrature: where the band ridge is and how wide it is, packaged as breakpoints for the harness. |
| `generate_events.py` | The independent physics simulator (truth generator) for the two simulator tests. |
| `sample_from_pdf.py` | Grid-based sampling from an arbitrary PDF (lintsampler), used by the self-consistency tests. |
| `quad_breakpoints_demo.py` | Self-contained demonstration (no repo dependencies) of why adaptive quadrature silently misses narrow peaks and how `points=` fixes it. |
| `debug_bin_example.py` | Example driver for `chisquare_harness.debug_bin`: point it at one (Ep, Eq) bin and a parameter set to see why that bin's expected count looks the way it does. |

## Integrating the PDF over bins

The expected count in a bin is the PDF integrated over the bin rectangle.
This is the subtlest part of the tests, because the band is a narrow ridge:
adaptive quadrature over a wide tail bin can sample only the flat ~zero
region, agree with itself that the integral is zero, and never subdivide
toward the peak — a silent wrong answer, not an error.

Suggested reading order:

1. `quad_breakpoints_demo.py` — run it and watch plain `quad` return ~0 for
   a narrow peak, then fix it with the `points=` argument.
2. `band_breakpoints.py` — where the ridge is physically: the noiseless
   band centroid Er → (Ep, Eq) from the yield model and Neganov-Luke
   amplification, inverted to give the most probable Eq at each Ep, plus
   the local band width setting the breakpoint window.
3. `chisquare_harness.py`, `expected_counts_from_pdf` — the nested
   quadrature that requires those breakpoints (`inner_points_func`) and
   normalizes the per-bin integrals into expected counts.

## Debugging a single bin

When a bin's expected count comes out at 0 or suspiciously small, it isn't
automatically a quadrature bug: a wide tail bin can genuinely sit mostly (or
entirely) off the band, in which case a tiny expected count is the correct
answer. `chisquare_harness.debug_bin(pdf_func, bin_rect, inner_points_func)`
tells the two cases apart by computing the same bin three independent ways:

- **naive** — nested quad with no breakpoints (what you'd get without
  `inner_points_func`; reproduces the missed-peak failure mode).
- **fixed** — nested quad with `inner_points_func`, exactly as
  `expected_counts_from_pdf` computes it.
- **grid_ref** — a brute-force fixed-grid Simpson integral, independent of
  quad's adaptive logic entirely.

It also prints where `inner_points_func`'s breakpoints land relative to the
bin at a few Ep values, flagging when all of them fall outside `[ymin,
ymax]` — a sign the ridge has drifted off the bin's Eq range there rather
than quad having missed anything.

If all three numbers agree, the small value is real (the bin is genuinely
mostly off the ridge). If `naive` disagrees with `fixed`/`grid_ref`, that's
the silent-miss failure from the "Integrating the PDF over bins" section
above, on this specific bin.

`debug_bin_example.py` is a ready-to-edit driver: set `BIN` and the
parameter dict to the case you're chasing and run

```
LD_LIBRARY_PATH=lib python test/python/debug_bin_example.py
```
