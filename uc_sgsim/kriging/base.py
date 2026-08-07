from __future__ import annotations

import warnings

import numpy as np
from uc_sgsim.cov_model.base import CovModel


class Kriging:
    """
    Base Class for Kriging Interpolation

    This class serves as the parent class for various kriging interpolation techniques
    such as Simple Kriging and Ordinary Kriging. It define and encapsulates common properties and
    methods used in kriging.

    Attributes:
        model (CovModel): The covariance model used for interpolation.
        bandwidth_step (int): The step size for bandwidth increments.
        bandwidth (np.array): An array of bandwidth values.
        k_range (float): The range parameter for the covariance model.
        sill (float): The sill parameter for the covariance model.
        x_size (int): Size of the x-axis for the interpolation grid.
        y_size (int): Size of the y-axis for the interpolation grid (0 if 1D).
        _cov_cache_flag (bool): Flag indicating whether to use a covariance cache.
        _cov_cache (list | list[list]): Cache for computed covariances (if enabled).

    Methods:
        _create_cov_cache(): Create the covariance cache for faster computations.
    """

    def __init__(
        self,
        model: CovModel,
        grid_size: int | list[int, int],
        cov_cache: bool = False,
        mean: float = 0.0,
    ):
        if not np.isfinite(mean):
            raise ValueError('mean must be finite')

        self._model = model
        self._bandwidth_step = model.bandwidth_step
        self._bandwidth = model.bandwidth
        self._k_range = model.k_range
        self._sill = model.sill
        self._mean = float(mean)
        self.x_size = grid_size if isinstance(grid_size, int) else grid_size[0]
        self.y_size = 0 if isinstance(grid_size, int) else grid_size[1]
        self._cov_cache_flag = cov_cache
        if cov_cache is True:
            self._cov_cache = {}

    @property
    def model(self) -> CovModel:
        return self._model

    @property
    def bandwidth_step(self) -> int:
        return self._bandwidth_step

    @property
    def bandwidth(self) -> np.array:
        return self._bandwidth

    @property
    def k_range(self) -> float:
        return self._k_range

    @property
    def sill(self) -> float:
        return self._sill

    @property
    def mean(self) -> float:
        return self._mean

    @staticmethod
    def _standard_normal(
        normal_score: float | None = None,
        rng: np.random.Generator | None = None,
    ) -> float:
        if normal_score is None:
            normal_score = np.random.normal() if rng is None else rng.normal()
        normal_score = float(normal_score)
        if not np.isfinite(normal_score):
            raise ValueError('normal_score must be finite')
        return normal_score

    def _unconditional_simulation(
        self,
        normal_score: float | None = None,
        rng: np.random.Generator | None = None,
    ) -> float:
        score = self._standard_normal(normal_score=normal_score, rng=rng)
        return float(self.mean + np.sqrt(self.sill) * score)

    def _solve_system(
        self,
        matrix: np.ndarray,
        vector: np.ndarray,
        covariance_size: int | None = None,
    ) -> np.ndarray:
        """Solve a kriging system, adding only scale-aware fallback jitter."""
        try:
            return np.linalg.solve(matrix, vector)
        except np.linalg.LinAlgError as original_error:
            covariance_size = matrix.shape[0] if covariance_size is None else covariance_size
            diagonal = np.diag(matrix[:covariance_size, :covariance_size])
            scale = max(float(np.max(np.abs(diagonal))), np.finfo(float).tiny)
            jitter = scale * np.finfo(float).eps * max(covariance_size, 1) * 16

            for _ in range(6):
                regularized = matrix.copy()
                indices = np.arange(covariance_size)
                regularized[indices, indices] += jitter
                try:
                    solution = np.linalg.solve(regularized, vector)
                except np.linalg.LinAlgError:
                    jitter *= 10
                    continue

                warnings.warn(
                    'Kriging covariance matrix was singular. '
                    'Added diagonal jitter {:.3e}.'.format(jitter),
                    RuntimeWarning,
                    stacklevel=2,
                )
                return solution

            raise np.linalg.LinAlgError(
                'Kriging covariance system is singular after adaptive diagonal jitter. '
                'Check for duplicate coordinates or an invalid covariance model.',
            ) from original_error

    def _standard_deviation(self, variance: float) -> float:
        variance = float(variance)
        if not np.isfinite(variance):
            raise FloatingPointError('Kriging variance is not finite')

        tolerance = max(abs(self.sill), np.finfo(float).tiny) * 1e-10
        if variance < -tolerance:
            raise FloatingPointError(
                'Kriging variance is negative ({:.6e}). '
                'The covariance system may not be positive semidefinite.'.format(variance),
            )
        return float(np.sqrt(max(variance, 0.0)))

    def __repr__(self):
        return f'{self.__class__.__name__}'
