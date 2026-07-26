from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist, squareform
from uc_sgsim.kriging.base import Kriging


class SimpleKriging(Kriging):
    """
    Simple Kriging Class

    This class represents the Simple Kriging interpolation technique, which is used
    to estimate values at unsampled locations based on sampled data and a specified
    covariance model.

    Attributes:
        model (CovModel): The covariance model used for interpolation.
        grid_size (int | list[int, int]): Size of the grid for interpolation.
        cov_cache (bool): Flag indicating whether to use a covariance cache.

    Methods:
        prediction(sample: np.array, unsampled: np.array) -> tuple[float, float]:
            Perform Simple Kriging prediction for an unsampled location.
        simulation(x: np.array, unsampled: np.array, **kwargs) -> float:
            Perform Simple Kriging simulation for unsampled locations.
        _find_neighbor(dist: list[float], neighbor: int) -> float:
            Find a nearby point for simulation based on a neighbor criterion.
    """

    def prediction(
        self,
        unsampled: np.ndarray,
        sampled: np.ndarray,
        dist_diff: np.ndarray = None,
    ) -> tuple[float, float]:
        """
        Perform Simple Kriging prediction for an unsampled location.

        Args:
            unsampled (np.array): Unsamped location for which prediction is made.
            sample (np.array): Sampled data for neighboring locations.

        Returns:
            tuple[float, float]: Estimated value and kriging standard deviation.
        """
        sampled = np.asarray(sampled, dtype=float)
        if sampled.ndim != 2 or sampled.shape[0] == 0 or sampled.shape[1] < 3:
            raise ValueError('sampled must contain rows of [x, y, value]')

        dist_diff = (
            np.asarray(dist_diff, dtype=float)
            if dist_diff is not None
            else np.linalg.norm(unsampled - sampled[:, [0, 1]], axis=1).flatten()
        )

        if self._cov_cache_flag is True:
            cov_dist = np.asarray(self.model.cov_compute(dist_diff), dtype=float)
            self._cov_cache[f'{unsampled[0]}, {unsampled[1]}'] = cov_dist
        elif hasattr(self, '_cov_cache') is True:
            cov_dist = self._cov_cache[f'{unsampled[0]}, {unsampled[1]}']
        else:
            cov_dist = np.asarray(self.model.cov_compute(dist_diff), dtype=float)

        pairwise_distances = squareform(pdist(sampled[:, :2]))
        cov_data = np.asarray(self.model.cov_compute(pairwise_distances), dtype=float)

        weights = self._solve_system(cov_data, cov_dist)
        residuals = sampled[:, 2] - self.mean
        estimation = self.mean + float(np.dot(weights, residuals))
        kriging_var = self.model.sill - float(np.dot(weights, cov_dist))

        return float(estimation), self._standard_deviation(kriging_var)

    def simulation(self, unsampled: np.ndarray, sampled: np.ndarray, **kwargs) -> float:
        """
        Perform Simple Kriging simulation for unsampled locations.

        Args:
            unsampled (np.ndarray): Sampled data for neighboring locations.
            unsampled (np.ndarray): Unsamped location for which simulation is performed.
            neighbor (int): The number of neighbors to consider (optional).

        Returns:
            float: Simulated value for the unsampled location.
        """
        normal_score = kwargs.get('normal_score')
        rng = kwargs.get('rng')
        if len(sampled) == 0:
            return self._unconditional_simulation(normal_score=normal_score, rng=rng)

        neighbor = kwargs.get('neighbor')
        if neighbor is not None:
            if not isinstance(neighbor, (int, np.integer)) or neighbor < 0:
                raise ValueError('neighbor must be a non-negative integer')
            distances = np.linalg.norm(unsampled - sampled[:, [0, 1]], axis=1)

            draw_random_normal = self._find_neighbor(
                distances,
                neighbor,
                normal_score=normal_score,
                rng=rng,
            )
            if draw_random_normal is not None:
                return draw_random_normal

            sorted_indices = np.argsort(distances)
            sampled = np.array(sampled)[sorted_indices][:neighbor]
            distances = distances[sorted_indices][:neighbor]

        dist_diff = distances if neighbor is not None else None
        estimation, kriging_std = self.prediction(unsampled, sampled, dist_diff)

        score = self._standard_normal(normal_score=normal_score, rng=rng)
        return float(estimation + kriging_std * score)

    def _find_neighbor(
        self,
        distances: list[float],
        neighbor: int,
        normal_score: float | None = None,
        rng: np.random.Generator | None = None,
    ) -> float | None:
        """
        Draw unconditionally only when neighborhood conditioning is disabled.

        Args:
            distances (list[float]): Distances from sampled points to the target.
            neighbor (int): The number of neighbors to consider.

        Returns:
            An unconditional draw when ``neighbor`` is zero; otherwise ``None``.

        Notes:
            Gaussian and exponential covariance remains non-zero beyond their
            practical range, so distance alone must not silently turn a
            conditional draw into an independent one.
        """
        if neighbor == 0:
            return self._unconditional_simulation(normal_score=normal_score, rng=rng)
        return None


class OrdinaryKriging(SimpleKriging):
    """
    Ordinary Kriging Class

    This class represents the Ordinary Kriging interpolation technique, which is
    an extension of Simple Kriging. It is used to estimate values at unsampled
    locations based on sampled data and a specified covariance model.

    Attributes:
        model (CovModel): The covariance model used for interpolation.
        grid_size (int | list[int, int]): Size of the grid for interpolation.
        cov_cache (bool): Flag indicating whether to use a covariance cache.

    Methods:
        prediction(sample: np.array, unsampled: np.array) -> tuple[float, float]:
            Perform Ordinary Kriging prediction for an unsampled location.
        matrix_augmented(mat: np.array) -> np.array:
            Augment the covariance matrix for Ordinary Kriging.
    """

    def prediction(
        self,
        unsampled: np.ndarray,
        sampled: np.ndarray,
        dist_diff: np.ndarray = None,
    ) -> tuple[float, float]:
        """
        Perform Ordinary Kriging prediction for an unsampled location.

        Args:
            unsampled (np.array): Unsamped location for which prediction is made.
            sample (np.array): Sampled data for neighboring locations.

        Returns:
            tuple[float, float]: Estimated value and kriging standard deviation.
        """
        sampled = np.asarray(sampled, dtype=float)
        if sampled.ndim != 2 or sampled.shape[0] == 0 or sampled.shape[1] < 3:
            raise ValueError('sampled must contain rows of [x, y, value]')

        n_sampled = len(sampled)
        dist_diff = (
            np.asarray(dist_diff, dtype=float)
            if dist_diff is not None
            else np.linalg.norm(unsampled - sampled[:, [0, 1]], axis=1).flatten()
        )

        if self._cov_cache_flag:
            cov_dist = np.asarray(self.model.cov_compute(dist_diff), dtype=float)
            self._cov_cache[f'{unsampled[0]}, {unsampled[1]}'] = cov_dist
        elif hasattr(self, '_cov_cache'):
            cov_dist = self._cov_cache[f'{unsampled[0]}, {unsampled[1]}']
        else:
            cov_dist = np.asarray(self.model.cov_compute(dist_diff), dtype=float)

        pairwise_distances = squareform(pdist(sampled[:, :2]))
        cov_data = np.asarray(self.model.cov_compute(pairwise_distances), dtype=float)

        cov_data_augmented = self._matrix_augmented(cov_data)
        cov_dist_augmented = np.append(cov_dist, 1.0)
        solution = self._solve_system(
            cov_data_augmented,
            cov_dist_augmented,
            covariance_size=n_sampled,
        )
        weights = solution[:n_sampled]
        lagrange_multiplier = float(solution[-1])

        estimation = float(np.dot(weights, sampled[:, 2]))
        kriging_var = self.model.sill - float(np.dot(weights, cov_dist)) - lagrange_multiplier

        return estimation, self._standard_deviation(kriging_var)

    def _matrix_augmented(self, mat: np.ndarray) -> np.ndarray:
        """
        Augment the covariance matrix to constrain summation of weights = 1 for Ordinary Kriging.

        Args:
            mat (np.array): Covariance matrix.

        Returns:
            np.array: Augmented covariance matrix.
        """
        ones_column = np.ones((mat.shape[0], 1))
        cov_data_augmented = np.hstack([mat, ones_column])
        ones_row = np.ones((1, cov_data_augmented.shape[1]))
        ones_row[0][-1] = 0
        cov_data_augmented = np.vstack((cov_data_augmented, ones_row))
        return cov_data_augmented
