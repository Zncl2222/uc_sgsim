from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist


class CovModel:
    """
    Covariance Model for Sequential Gaussian Simulation.

    This class represents a covariance model used in Sequential Gaussian Simulation
    for geostatistical simulations. It defines the spatial correlation structure based on
    variogram modeling.

    Attributes:
        __bandwidth_len (float): The length of the bandwidth.
        __bandwidth_step (float): The step size for bandwidth increments.
        __bandwidth (np.array): An array of bandwidth values.
        __k_range (float): The range parameter for the covariance model.
        __sill (float): The sill parameter for the covariance model.
        __nugget (float): The nugget effect parameter for the covariance model.

    Methods:
        cov_compute(x: np.array) -> np.array: Compute covariance values.
        var_compute(x: np.array) -> np.array: Compute variogram values.
        variogram(x: np.array) -> np.array: Compute variogram for a given dataset.
        variogram_plot(fig: int = None): Plot theoretical variogram using SgsimPlot.
    """

    def __init__(
        self,
        bandwidth_len: float,
        bandwidth_step: float,
        k_range: float,
        sill: float = 1,
        nugget: float = 0,
    ):
        """
        Initialize a CovModel object.

        Args:
            bandwidth_len (float): The length of the bandwidth.
            bandwidth_step (float): The step size for bandwidth increments.
            k_range (float): Practical/effective range for Gaussian and
                exponential models, where correlation is ``exp(-3)``;
                exact finite range for the spherical model.
            sill (float, optional): The sill parameter for the covariance model (default is 1).
            nugget (float, optional): The nugget effect parameter for the covariance model
                                      (default is 0).
        """
        parameters = {
            'bandwidth_len': bandwidth_len,
            'bandwidth_step': bandwidth_step,
            'k_range': k_range,
            'sill': sill,
            'nugget': nugget,
        }
        if any(not np.isfinite(value) for value in parameters.values()):
            raise ValueError('covariance model parameters must be finite')
        if bandwidth_len <= 0:
            raise ValueError('bandwidth_len must be greater than zero')
        if bandwidth_step <= 0:
            raise ValueError('bandwidth_step must be greater than zero')
        if k_range <= 0:
            raise ValueError('k_range must be greater than zero')
        if sill <= 0:
            raise ValueError('sill must be greater than zero')
        if nugget < 0 or nugget > sill:
            raise ValueError('nugget must satisfy 0 <= nugget <= sill')

        self._bandwidth_len = bandwidth_len
        self._bandwidth_step = bandwidth_step
        self._bandwidth = np.arange(0, bandwidth_len, bandwidth_step)
        self._k_range = k_range
        self._sill = sill
        self._nugget = nugget

    @property
    def bandwidth_len(self) -> float:
        return self._bandwidth_len

    @property
    def bandwidth_step(self) -> float:
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
    def nugget(self) -> float:
        return self._nugget

    def __repr__(self) -> str:
        return (
            f'{self.model_name}Model(bandwidth_len={self.bandwidth_len}, '
            + f'bandwidth_step={self.bandwidth_step}, '
            + f'k_range={self.k_range}, sill={self.sill}, nugget={self.nugget})'
        )

    def cov_compute(self, x: np.ndarray | float) -> np.ndarray | float:
        """
        Compute covariance values for one or more lag distances.

        ``sill`` is the total point variance.  With a non-zero nugget,
        covariance is discontinuous at the origin:

        * ``C(0) = sill``
        * ``C(h) = (sill - nugget) * rho(h)`` for ``h > 0``

        Args:
            x: Non-negative lag distance or distances.

        Returns:
            Covariance value(s) with the same shape as ``x``.
        """
        lags = self._validate_lags(x)
        semivariogram = np.vectorize(self.model, otypes=[float])(lags)
        covariance = self._sill - semivariogram
        return float(covariance) if covariance.ndim == 0 else covariance

    def var_compute(self, x: np.ndarray | float) -> np.ndarray | float:
        """
        Compute theoretical semivariogram values.

        Args:
            x: Non-negative lag distance or distances.

        Returns:
            Semivariogram value(s) with the same shape as ``x``.
        """
        lags = self._validate_lags(x)
        semivariogram = np.vectorize(self.model, otypes=[float])(lags)
        return float(semivariogram) if semivariogram.ndim == 0 else semivariogram

    @staticmethod
    def _validate_lags(x: np.ndarray | float) -> np.ndarray:
        lags = np.asarray(x, dtype=float)
        if np.any(~np.isfinite(lags)):
            raise ValueError('lag distances must be finite')
        if np.any(lags < 0):
            raise ValueError('lag distances must be non-negative')
        return lags

    def variogram(self, x: np.array) -> np.array:
        """
        Compute variogram for a given dataset.

        Args:
            x (np.array): Input dataset for which the variogram is computed.

        Returns:
            np.array: Array of variogram values.
        """
        dist = cdist(x[:, :1], x[:, :1])
        variogram = []

        for h in self._bandwidth:
            indices = np.where(
                (dist >= h - self._bandwidth_step) & (dist <= h + self._bandwidth_step),
            )
            z = np.power(x[indices[0], 1] - x[indices[1], 1], 2)
            z_sum = np.sum(z)
            if z_sum >= 1e-7:
                variogram.append(z_sum / (2 * len(z)))

        return np.array(variogram)

    def variogram_plot(self, fig: int = None):
        """
        Plot the theoretical variogram using SgsimPlot.

        Args:
            fig (int, optional): The figure number to use for plotting (default is None).
        """
        from ..plotting.sgsim_plot import SgsimPlot

        SgsimPlot(model=self).theory_variogram_plot(fig=fig)
