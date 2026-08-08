from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from scipy.spatial.distance import cdist

import uc_sgsim as uc
from uc_sgsim.exception import IterationError


def _native_values(seed: int) -> np.ndarray:
    simulator = uc.SequentialGaussianSimulator(
        24,
        uc.Gaussian(12, 1, 6, sill=1.7, nugget=0.2),
        backend='c',
        max_neighbors=8,
    )
    return simulator.simulate(6, seed=seed).values


def test_native_engine_is_reentrant_for_concurrent_calls():
    expected = _native_values(2026)

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(_native_values, [2026] * 8))

    for result in results:
        np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize('model_class', [uc.Gaussian, uc.Exponential, uc.Spherical])
def test_native_ensemble_matches_covariance_contract(model_class):
    realization_count = 6000
    node_count = 5
    model = model_class(10, 1, 10, sill=1.7, nugget=0.2)
    simulator = uc.SequentialGaussianSimulator(
        node_count,
        model,
        backend='c',
        max_neighbors=node_count,
    )

    values = simulator.simulate(realization_count, seed=20260726).values[:, :, 0]
    coordinates = np.arange(node_count, dtype=float)[:, np.newaxis]
    expected_covariance = model.cov_compute(cdist(coordinates, coordinates))
    empirical_mean = values.mean(axis=0)
    empirical_covariance = np.cov(values, rowvar=False, ddof=1)
    mean_standard_error = np.sqrt(np.diag(expected_covariance) / realization_count)
    covariance_standard_error = np.sqrt(
        (
            expected_covariance**2
            + np.outer(np.diag(expected_covariance), np.diag(expected_covariance))
        )
        / (realization_count - 1),
    )

    assert np.all(np.abs(empirical_mean) <= 6 * mean_standard_error)
    assert np.all(
        np.abs(empirical_covariance - expected_covariance) <= 6 * covariance_standard_error,
    )


def test_native_covariance_cache_preserves_results():
    model = uc.Exponential(12, 1, 5, sill=1.7, nugget=0.2)
    options = {
        'grid_size': 24,
        'covariance': model,
        'backend': 'c',
        'constant_path': True,
        'max_neighbors': 8,
    }
    uncached = uc.SequentialGaussianSimulator(**options, covariance_cache=False)
    cached = uc.SequentialGaussianSimulator(**options, covariance_cache=True)

    expected = uncached.simulate(8, seed=321).values
    actual = cached.simulate(8, seed=321).values

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('backend', ['python', 'c'])
@pytest.mark.parametrize(
    ('nugget', 'max_neighbors'),
    [
        (0.2, 0),
        (1.7, 5),
    ],
)
def test_no_conditioning_and_pure_nugget_are_white_noise(
    backend,
    nugget,
    max_neighbors,
):
    realization_count = 6000
    node_count = 5
    sill = 1.7
    model = uc.Gaussian(10, 1, 10, sill=sill, nugget=nugget)
    simulator = uc.SequentialGaussianSimulator(
        node_count,
        model,
        backend=backend,
        max_neighbors=max_neighbors,
    )

    values = simulator.simulate(realization_count, seed=20260808).values[:, :, 0]
    expected_covariance = np.eye(node_count) * sill
    empirical_mean = values.mean(axis=0)
    empirical_covariance = np.cov(values, rowvar=False, ddof=1)
    mean_standard_error = np.sqrt(sill / realization_count)
    covariance_standard_error = np.sqrt(
        (
            expected_covariance**2
            + np.outer(np.diag(expected_covariance), np.diag(expected_covariance))
        )
        / (realization_count - 1),
    )

    assert np.all(np.abs(empirical_mean) <= 6 * mean_standard_error)
    assert np.all(
        np.abs(empirical_covariance - expected_covariance) <= 6 * covariance_standard_error,
    )


@pytest.mark.parametrize(
    ('model_class', 'k_range'),
    [
        (uc.Exponential, 3),
        (uc.Gaussian, 8),
    ],
)
@pytest.mark.parametrize('constant_path', [False, True])
def test_finite_neighborhood_path_statistics(model_class, k_range, constant_path):
    realization_count = 6000
    node_count = 12
    sill = 1.7
    model = model_class(12, 1, k_range, sill=sill, nugget=0.2)
    simulator = uc.SequentialGaussianSimulator(
        node_count,
        model,
        backend='c',
        max_neighbors=4,
        constant_path=constant_path,
    )

    values = simulator.simulate(
        realization_count,
        seed=20260808,
    ).values[:, :, 0]
    empirical_mean = values.mean(axis=0)
    centered = values - empirical_mean
    empirical_variance = np.sum(centered**2, axis=0) / (realization_count - 1)
    empirical_lag_one = np.sum(centered[:, :-1] * centered[:, 1:], axis=0) / (realization_count - 1)
    expected_lag_one = float(model.cov_compute(1.0))

    mean_standard_error = np.sqrt(sill / realization_count)
    variance_standard_error = np.sqrt(2 * sill**2 / (realization_count - 1))
    lag_standard_error = np.sqrt(
        (expected_lag_one**2 + sill**2) / (realization_count - 1),
    )

    assert np.all(np.abs(empirical_mean) <= 5 * mean_standard_error)
    assert np.all(np.abs(empirical_variance - sill) <= 5 * variance_standard_error)
    assert np.all(
        np.abs(empirical_lag_one - expected_lag_one) <= 5 * lag_standard_error,
    )


def test_native_iteration_limit_is_visible_to_python():
    simulator = uc.SequentialGaussianSimulator(
        8,
        uc.Gaussian(8, 1, 4),
        backend='c',
        iteration_limit=1,
        min_value=-0.001,
        max_value=0.001,
    )

    with pytest.raises(IterationError, match='rejection limit'):
        simulator.simulate(1, seed=7)


def test_native_simulation_has_no_hidden_file_output(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    simulator = uc.SequentialGaussianSimulator(
        8,
        uc.Gaussian(8, 1, 4),
        backend='c',
    )

    simulator.simulate(2, seed=7)

    assert list(tmp_path.iterdir()) == []
