# Python API v2

The result-oriented API separates reusable simulation configuration from one
execution and its output:

```python
from uc_sgsim import Gaussian, SequentialGaussianSimulator

covariance = Gaussian(
    bandwidth_len=35,
    bandwidth_step=1,
    k_range=17.32,
    sill=1.0,
)

simulator = SequentialGaussianSimulator(
    grid_size=151,
    covariance=covariance,
    kriging="simple",
    mean=0.0,
    max_neighbors=8,
    backend="python",
)

result = simulator.simulate(
    n_realizations=10,
    seed=151,
    workers=4,
)

values = result.values
mutable_values = result.to_numpy()
```

`result.values` always has the shape `(realizations, x, y)`, including for a
1D grid where `y == 1`. The array is read-only so later application code cannot
silently change a completed result. `to_numpy()` returns a mutable copy.

`n_realizations` is the total number returned, independent of `workers`. The
legacy engine uses equal-sized worker batches; when the total is not divisible
by the worker count, the facade generates and trims at most `workers - 1` extra
realizations. A worker count larger than the requested realization count is
capped automatically.

## Compatibility

`UCSgsim` remains available with its existing constructor, mutable attributes,
plotting methods, saving methods, and worker-count behavior. Existing programs
can migrate one call site at a time.

The new facade delegates to `UCSgsim`, so this API change does not alter the
validated covariance, kriging, Python simulation, or C simulation formulas.

The C backend currently supports one-dimensional grids, Gaussian, exponential,
and spherical covariance, Simple Kriging, and a zero mean. The facade rejects
unsupported combinations rather than silently selecting a different
implementation.

Ordinary Kriging remains available as a standalone interpolation estimator,
but the unconditional simulator rejects it. Its sum-to-one constraint does not
produce the conditional factorization of the configured stationary Gaussian
field.
