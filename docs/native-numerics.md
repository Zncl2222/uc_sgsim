# Native numerical core

The native engine owns its foundational array and matrix code. It no longer
depends on the experimental `c_array_tools` submodule.

## Module boundaries

- `native_array.h` provides fixed-size `int` and `double` buffers. Allocation
  is zero-initialized, size multiplication is checked before allocation, and
  release operations are NULL-safe and idempotent.
- `native_matrix.h` provides contiguous row-major matrices, bounds-checked
  accessors, regularized Cholesky factorization, triangular solve, dot product,
  and a symmetric quadratic form.
- `random_tools.h` owns the deterministic MT19937 state used by the simulator.
  Moving it into the project removes the array-library coupling while
  preserving the existing seed-to-result sequence.

The simulation hot path accesses contiguous storage directly only after the
workspace dimensions have been validated. Public helpers retain checked
accessors for code that does not have that invariant.

## Ownership contract

An initialized array or matrix has exactly one owner. Initialize fresh or
zeroed structures, release them with their matching `*_free` function, and do
not copy owning structures by value. The engine performs all workspace
allocation before entering the simulation loop, so Cholesky factorization and
Kriging solve do not allocate memory.

Zero-length containers are valid and own no storage. Overflow, allocation
failure, invalid dimensions, non-finite input, and non-positive-definite
matrices are reported with `sgsim_numeric_status_t` rather than terminating the
process.

## Numerical and performance choices

- Matrices use one row-major allocation instead of per-row allocations. This
  improves locality and makes destruction constant-time.
- The covariance solver uses Cholesky factorization because covariance
  matrices are symmetric positive definite. A failed factorization is retried
  with bounded, scale-aware diagonal jitter.
- The quadratic form reads one triangle and mirrors off-diagonal contributions,
  roughly halving its matrix reads.
- Experimental variograms iterate directly over integer lags. They require
  O(1) auxiliary memory instead of materializing an O(n²) distance matrix.
- Variance uses Welford's online update, avoiding `pow` and reducing
  cancellation for data with a large offset.

## Validation

The native test target covers allocation overflow, checked bounds, idempotent
release, Cholesky solve accuracy, non-finite rejection, regularization, legacy
LU compatibility, variogram windows, and simulator integration. CI also runs
the full Python suite, ASan/UBSan, and Valgrind. For reproducible engine timing:

```bash
python benchmarks/native_engine.py --grid-size 400 --realizations 250 \
    --neighbors 12 --repeats 5
```
