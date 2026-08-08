import numpy as np
import pytest
from scipy.spatial.distance import cdist

from uc_sgsim import UCSgsim
from uc_sgsim.cov_model import Exponential, Gaussian, Spherical
from uc_sgsim.kriging import OrdinaryKriging, SimpleKriging


@pytest.mark.scientific
class TestCovarianceContract:
    @pytest.mark.parametrize(
        ('model_class', 'expected_at_range'),
        [
            (Gaussian, 0.8 * np.exp(-3.0)),
            (Exponential, 0.8 * np.exp(-3.0)),
            (Spherical, 0.0),
        ],
    )
    def test_nugget_is_discontinuous_only_at_origin(self, model_class, expected_at_range):
        model = model_class(10, 1, 3, sill=1.0, nugget=0.2)

        covariance = model.cov_compute(np.array([0.0, 3.0]))
        semivariogram = model.var_compute(np.array([0.0, 3.0]))

        assert covariance[0] == pytest.approx(1.0)
        assert semivariogram[0] == pytest.approx(0.0)
        assert covariance[1] == pytest.approx(expected_at_range)
        assert semivariogram[1] == pytest.approx(1.0 - expected_at_range)

    @pytest.mark.parametrize(
        'kwargs',
        [
            {'bandwidth_len': 0, 'bandwidth_step': 1, 'k_range': 3},
            {'bandwidth_len': 10, 'bandwidth_step': 0, 'k_range': 3},
            {'bandwidth_len': 10, 'bandwidth_step': 1, 'k_range': 0},
            {'bandwidth_len': 10, 'bandwidth_step': 1, 'k_range': 3, 'sill': 0},
            {
                'bandwidth_len': 10,
                'bandwidth_step': 1,
                'k_range': 3,
                'sill': 1,
                'nugget': -0.1,
            },
            {
                'bandwidth_len': 10,
                'bandwidth_step': 1,
                'k_range': 3,
                'sill': 1,
                'nugget': 1.1,
            },
            {'bandwidth_len': 10, 'bandwidth_step': 1, 'k_range': np.inf},
        ],
    )
    def test_invalid_covariance_parameters_are_rejected(self, kwargs):
        with pytest.raises(ValueError):
            Gaussian(**kwargs)

    def test_negative_and_non_finite_lags_are_rejected(self):
        model = Gaussian(10, 1, 3)

        with pytest.raises(ValueError, match='non-negative'):
            model.cov_compute(np.array([-1.0]))
        with pytest.raises(ValueError, match='finite'):
            model.var_compute(np.array([np.inf]))


@pytest.mark.scientific
class TestConditionalGaussianOracle:
    def test_simple_kriging_matches_block_gaussian_conditioning(self):
        model = Gaussian(10, 1, 4, sill=1.7, nugget=0.2)
        mean = 2.5
        kriging = SimpleKriging(model, 3, mean=mean)
        target = np.array([1.0, 0.0])
        sampled = np.array(
            [
                [0.0, 0.0, 1.2, 99.0],
                [2.0, 0.0, 3.4, 99.0],
            ],
        )
        sampled_before = sampled.copy()

        sample_coordinates = sampled[:, :2]
        covariance = model.cov_compute(cdist(sample_coordinates, sample_coordinates))
        cross_covariance = model.cov_compute(cdist(sample_coordinates, target[None, :])).ravel()
        weights = np.linalg.solve(covariance, cross_covariance)
        expected_mean = mean + weights @ (sampled[:, 2] - mean)
        expected_variance = model.sill - weights @ cross_covariance

        estimate, standard_deviation = kriging.prediction(target, sampled)

        assert estimate == pytest.approx(expected_mean, abs=1e-13)
        assert standard_deviation**2 == pytest.approx(expected_variance, abs=1e-13)
        np.testing.assert_array_equal(sampled, sampled_before)

    def test_jittered_simple_kriging_uses_general_error_variance(self):
        model = Gaussian(10, 1, 4, sill=1.7, nugget=0.2)
        kriging = SimpleKriging(model, 3)
        target = np.array([1.0, 0.0])
        sampled = np.array(
            [
                [0.0, 0.0, 1.2, 0.0],
                [0.0, 0.0, 1.2, 0.0],
            ],
        )
        covariance = model.cov_compute(cdist(sampled[:, :2], sampled[:, :2]))
        cross_covariance = model.cov_compute(
            cdist(sampled[:, :2], target[None, :]),
        ).ravel()

        with pytest.warns(RuntimeWarning, match='diagonal jitter'):
            _, standard_deviation = kriging.prediction(target, sampled)

        regularized = covariance + np.eye(2) * kriging._last_diagonal_jitter
        weights = np.linalg.solve(regularized, cross_covariance)
        expected_variance = (
            model.sill - 2.0 * weights @ cross_covariance + weights @ covariance @ weights
        )
        assert kriging._last_diagonal_jitter > 0.0
        assert standard_deviation**2 == pytest.approx(expected_variance, abs=1e-13)

    def test_ordinary_kriging_variance_includes_lagrange_multiplier(self):
        model = Gaussian(10, 1, 3, sill=1.0)
        kriging = OrdinaryKriging(model, 2)
        target = np.array([1.0, 0.0])
        sampled = np.array([[0.0, 0.0, 7.0, 0.0]])
        cross_covariance = float(model.cov_compute(1.0))

        estimate, standard_deviation = kriging.prediction(target, sampled)

        assert estimate == pytest.approx(7.0)
        assert standard_deviation**2 == pytest.approx(
            2 * (model.sill - cross_covariance),
            abs=1e-13,
        )

    def test_full_neighborhood_sgs_matches_cholesky_for_fixed_scores(self):
        node_count = 5
        coordinates = np.column_stack([np.arange(node_count), np.zeros(node_count)])
        model = Gaussian(10, 1, 10, sill=1.7, nugget=0.2)
        mean = 2.5
        scores = np.array([0.2, -1.1, 0.7, 0.3, -0.4])
        covariance = model.cov_compute(cdist(coordinates, coordinates))
        expected = mean + np.linalg.cholesky(covariance) @ scores
        kriging = SimpleKriging(model, node_count, mean=mean)

        sampled = np.empty((0, 4), dtype=float)
        simulated = np.empty(node_count)
        for index, coordinate in enumerate(coordinates):
            simulated[index] = kriging.simulation(
                coordinate,
                sampled,
                neighbor=index,
                normal_score=scores[index],
            )
            sampled = np.vstack(
                [sampled, [coordinate[0], coordinate[1], simulated[index], 0.0]],
            )

        np.testing.assert_allclose(simulated, expected, rtol=0.0, atol=2e-12)

    def test_nonzero_covariance_beyond_practical_range_is_not_discarded(self):
        model = Exponential(20, 1, 2, sill=1.0)
        mean = 1.5
        kriging = SimpleKriging(model, 2, mean=mean)
        target = np.array([3.0, 0.0])
        sampled = np.array([[0.0, 0.0, 11.5, 0.0]])
        cross_covariance = float(model.cov_compute(3.0))
        expected_conditional_mean = mean + cross_covariance * (sampled[0, 2] - mean)

        simulated = kriging.simulation(
            target,
            sampled,
            neighbor=1,
            normal_score=0.0,
        )

        assert cross_covariance > 0.0
        assert simulated == pytest.approx(expected_conditional_mean, abs=1e-13)
        assert simulated != pytest.approx(mean)

    def test_ensemble_moments_are_within_gaussian_sampling_uncertainty(self):
        realization_count = 5000
        node_count = 5
        coordinates = np.column_stack([np.arange(node_count), np.zeros(node_count)])
        model = Gaussian(10, 1, 10, sill=1.7, nugget=0.2)
        mean = 2.5
        covariance = model.cov_compute(cdist(coordinates, coordinates))
        kriging = SimpleKriging(model, node_count, mean=mean)
        rng = np.random.default_rng(20260726)
        realizations = np.empty((realization_count, node_count))

        for realization_index in range(realization_count):
            sampled = np.empty((0, 4), dtype=float)
            for node_index, coordinate in enumerate(coordinates):
                value = kriging.simulation(
                    coordinate,
                    sampled,
                    neighbor=node_index,
                    rng=rng,
                )
                realizations[realization_index, node_index] = value
                sampled = np.vstack(
                    [sampled, [coordinate[0], coordinate[1], value, 0.0]],
                )

        empirical_mean = realizations.mean(axis=0)
        empirical_covariance = np.cov(realizations, rowvar=False, ddof=1)
        mean_standard_error = np.sqrt(np.diag(covariance) / realization_count)
        covariance_standard_error = np.sqrt(
            (covariance**2 + np.outer(np.diag(covariance), np.diag(covariance)))
            / (realization_count - 1),
        )

        assert np.all(np.abs(empirical_mean - mean) <= 5 * mean_standard_error)
        assert np.all(
            np.abs(empirical_covariance - covariance) <= 5 * covariance_standard_error,
        )


@pytest.mark.scientific
class TestSimulationContract:
    @pytest.mark.parametrize(
        'kriging',
        ['OrdinaryKriging', OrdinaryKriging(Gaussian(10, 1, 3), 5)],
    )
    def test_unconditional_simulation_rejects_ordinary_kriging(self, kriging):
        model = Gaussian(10, 1, 3)

        with pytest.raises(ValueError, match='unconditional stationary SGS'):
            UCSgsim(5, 1, model, kriging=kriging)

    def test_default_simulation_is_not_tail_truncated(self):
        model = Gaussian(10, 1, 3)
        simulation = UCSgsim(5, 1, model, mean=3.0)

        assert np.isneginf(simulation.min_value)
        assert np.isposinf(simulation.max_value)
        assert simulation.mean == pytest.approx(3.0)
        assert simulation.kriging.mean == pytest.approx(3.0)

    def test_invalid_bounds_are_rejected(self):
        model = Gaussian(10, 1, 3)

        with pytest.raises(ValueError, match='min_value'):
            UCSgsim(5, 1, model, min_value=1.0, max_value=1.0)
        with pytest.raises(ValueError, match='NaN'):
            UCSgsim(5, 1, model, min_value=np.nan)

    def test_invalid_neighbor_count_is_rejected(self):
        model = Gaussian(10, 1, 3)

        with pytest.raises(ValueError, match='max_neighbor'):
            UCSgsim(5, 1, model, max_neighbor=1.5)

    def test_python_engine_runs_with_unbounded_defaults(self):
        model = Gaussian(10, 1, 5, sill=1.2, nugget=0.1)
        simulation = UCSgsim(5, 3, model, mean=2.0, max_neighbor=5)

        simulation.run(n_processes=1, randomseed=1234)

        assert simulation.random_fields.shape == (3, 5, 1)
        assert np.all(np.isfinite(simulation.random_fields))

    def test_materially_negative_variance_raises_diagnostic(self):
        model = Gaussian(10, 1, 3)
        kriging = SimpleKriging(model, 2)

        with pytest.raises(FloatingPointError, match='negative'):
            kriging._standard_deviation(-1e-3)
