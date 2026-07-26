import numpy as np
from uc_sgsim.cov_model.base import CovModel


class Gaussian(CovModel):
    model_name = 'Gaussian'

    def model(self, h: float) -> float:
        """
        Compute the Gaussian semivariogram for a given lag distance (h).

        Args:
            h (float): Lag distance at which to compute the covariance.

        Returns:
            float: The computed Gaussian semivariogram value.
        """
        if h == 0:
            return 0.0
        partial_sill = self.sill - self.nugget
        return float(
            self.nugget + partial_sill * (1 - np.exp(-3 * h**2 / self.k_range**2)),
        )


class Spherical(CovModel):
    model_name = 'Spherical'

    def model(self, h: float) -> float:
        """
        Compute the Spherical semivariogram for a given lag distance (h).

        Args:
            h (float): Lag distance at which to compute the covariance.

        Returns:
            float: The computed Spherical semivariogram value.
        """
        if h == 0:
            return 0.0
        partial_sill = self.sill - self.nugget
        if h <= self.k_range:
            return float(
                partial_sill * (1.5 * h / self.k_range - 0.5 * (h / self.k_range) ** 3.0)
                + self.nugget
            )
        return float(self.sill)


class Exponential(CovModel):
    model_name = 'Exponential'

    def model(self, h: float) -> float:
        """
        Compute the Exponential semivariogram for a given lag distance (h).

        Args:
            h (float): Lag distance at which to compute the covariance.

        Returns:
            float: The computed Exponential semivariogram value.
        """
        if h == 0:
            return 0.0
        partial_sill = self.sill - self.nugget
        return float(
            self.nugget + partial_sill * (1 - np.exp(-3 * h / self.k_range)),
        )
