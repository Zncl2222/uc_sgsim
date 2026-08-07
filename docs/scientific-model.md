# Scientific model and validation contract

This document defines the probability model that the Python reference backend
must implement. It also defines the tests that future native backends must pass
before claiming scientific parity.

## Scope

The reference algorithm is unconditional Sequential Gaussian Simulation (SGS)
with a stationary Gaussian random field and Simple Kriging (SK).
"Unconditional" means that no observed hard data are supplied initially. It
does not mean that nodes are independent: every previously simulated node may
condition the next node.

Ordinary Kriging (OK) remains available as a practical unknown-mean variant,
but it is not the exact conditional factorization of a fixed-mean Gaussian
field. The legacy C backend is retained for compatibility and is not currently
part of this scientific reference contract.

## Covariance and semivariogram contract

Let

- \(\sigma^2\) be `sill`, the total variance at one grid node;
- \(\tau^2\) be `nugget`;
- \(a\) be `k_range`;
- \(\rho(h)\) be the spatial correlation for positive lag \(h\).

The covariance is

$$
C(0)=\sigma^2,
\qquad
C(h)=(\sigma^2-\tau^2)\rho(h),\quad h>0.
$$

The corresponding semivariogram is

$$
\gamma(0)=0,
\qquad
\gamma(h)=\tau^2+(\sigma^2-\tau^2)(1-\rho(h)),\quad h>0.
$$

This discontinuity at the origin is the nugget effect. The implementation uses
the following isotropic correlation functions:

$$
\rho_G(h)=\exp\left[-3\left(\frac{h}{a}\right)^2\right],
$$

$$
\rho_E(h)=\exp\left(-3\frac{h}{a}\right),
$$

and

$$
\rho_S(h)=
\begin{cases}
1-\frac{3h}{2a}+\frac{h^3}{2a^3}, & 0<h\le a,\\
0, & h>a.
\end{cases}
$$

For Gaussian and exponential models, `k_range` is the practical range because
the correlation at \(h=a\) is \(e^{-3}\), approximately 0.05. For the spherical
model, it is the exact finite range.

The constructor rejects non-positive sill or range, negative nugget, nugget
greater than sill, and invalid lag distances.

## Exact Simple Kriging conditional

For a target node \(x\), previously simulated coordinates \(A\), simulated
values \(z_A\), and known global mean \(m\), define

$$
K=C(A,A), \qquad k=C(A,x).
$$

The conditional Gaussian moments are

$$
\lambda=K^{-1}k,
$$

$$
\mu_c=m+\lambda^\mathsf{T}(z_A-m),
$$

$$
\sigma_c^2=C(0)-\lambda^\mathsf{T}k.
$$

One independent standard-normal score then produces

$$
z(x)=\mu_c+\sigma_c\epsilon,\qquad \epsilon\sim\mathcal N(0,1).
$$

Applying this conditional distribution sequentially follows the probability
chain rule. With a full neighborhood it samples the same joint distribution
as

$$
z=m+L\epsilon,\qquad LL^\mathsf{T}=C,
$$

where \(L\) is the lower Cholesky factor.

## Ordinary Kriging variance

For the covariance-form OK system

$$
\begin{bmatrix}
K & \mathbf 1\\
\mathbf 1^\mathsf{T} & 0
\end{bmatrix}
\begin{bmatrix}
\lambda\\
\nu
\end{bmatrix}
=
\begin{bmatrix}
k\\
1
\end{bmatrix},
$$

the error variance is

$$
\sigma_{\mathrm{OK}}^2=C(0)-\lambda^\mathsf{T}k-\nu.
$$

The Lagrange multiplier term must not be omitted.

## Numerical policy

- Kriging values and simulated values are scalar Python floats.
- The covariance matrix is solved without unconditional regularization.
- If the solve reports a singular matrix, a scale-aware diagonal jitter starts
  near machine precision and increases only as needed. A runtime warning reports
  the actual jitter.
- A negative conditional variance is clipped only when its magnitude is within
  \(10^{-10}\) of the sill scale. A more negative result raises a diagnostic
  error instead of silently changing the distribution.
- The default field is unbounded. Explicit `min_value` or `max_value` performs
  whole-realization rejection, which samples the Gaussian field conditioned on
  every node satisfying those limits. It is not ordinary unbounded SGS.

## Approximation boundaries

`max_neighbor` limits the conditioning set and is therefore a computational
approximation. It can be appropriate for large grids, especially when omitted
covariances are negligible, but exact small-grid validation must include every
previously simulated node. Selected Gaussian and exponential neighbors are not
silently discarded beyond their practical range because those covariance
functions remain non-zero there.

Random paths are useful with truncated neighborhoods to reduce directional
artifacts. For a full conditioning set, any fixed permutation is a valid
factorization of the same joint Gaussian distribution.

## Executable validation

`tests/scientific_model_test.py` verifies:

1. covariance and semivariogram behavior at the nugget discontinuity;
2. SK conditional mean and variance against independent block-Gaussian algebra;
3. OK variance including the Lagrange multiplier;
4. fixed-score, full-neighborhood SGS against direct Cholesky sampling;
5. ensemble mean and covariance against five-standard-error Gaussian sampling
   bounds using a fixed random seed;
6. retention of non-zero covariance beyond a practical range;
7. unbounded defaults and negative-variance diagnostics.

Run the scientific contract with:

```bash
pytest -q tests/scientific_model_test.py
```

## References

- Gómez-Hernández, *A gentle introduction to Sequential Gaussian Simulation*:
  <https://jgomez.webs.upv.es/wordpress/wp-content/uploads/2021/12/SGS.pdf>
- Gómez-Hernández and Journel, *Joint Sequential Simulation of Multi-Gaussian
  Fields*: <https://link.springer.com/chapter/10.1007/978-94-011-1739-5_8>
- Remy, Boucher, and Wu, *Applied Geostatistics with SGeMS*:
  <https://pangea.stanford.edu/departments/ere/dropbox/scrf/documents/reports/20/SCRF2007_Report20/SCRF2007_RemyBoucherWu_Book.pdf>
