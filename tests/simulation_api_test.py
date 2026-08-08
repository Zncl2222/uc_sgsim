import numpy as np
import pytest

import uc_sgsim as uc


@pytest.fixture
def covariance():
    return uc.Gaussian(
        bandwidth_len=4,
        bandwidth_step=1,
        k_range=3,
        sill=1,
    )


def test_new_api_matches_legacy_engine_for_one_worker(covariance):
    legacy = uc.UCSgsim(
        grid_size=5,
        realization_number=2,
        model=covariance,
        mean=0.25,
    )
    legacy.run(n_processes=1, randomseed=42)

    simulator = uc.SequentialGaussianSimulator(
        grid_size=5,
        covariance=covariance,
        mean=0.25,
    )
    result = simulator.simulate(n_realizations=2, seed=42)

    np.testing.assert_array_equal(result.values, legacy.random_fields)
    assert simulator.last_result is result
    assert result.shape == (2, 5, 1)
    assert result.n_realizations == 2
    assert result.grid_shape == (5, 1)
    assert result.seed == 42
    assert result.workers == 1
    assert result.backend == 'python'
    assert result.kriging == 'SimpleKriging'


def test_result_values_are_read_only_but_can_be_copied(covariance):
    result = uc.SequentialGaussianSimulator(4, covariance).simulate(1, seed=7)

    with pytest.raises(ValueError):
        result.values[0, 0, 0] = 99

    mutable_values = result.to_numpy()
    mutable_values[0, 0, 0] = 99

    assert mutable_values[0, 0, 0] != result.values[0, 0, 0]
    assert result.to_numpy(copy=False) is result.values
    np.testing.assert_array_equal(np.asarray(result), result.values)


def test_requested_count_is_independent_of_worker_count(covariance):
    simulator = uc.SequentialGaussianSimulator(4, covariance)

    result = simulator.simulate(n_realizations=3, seed=11, workers=2)

    assert result.shape == (3, 4, 1)
    assert result.workers == 2


def test_worker_count_is_capped_to_useful_processes(covariance):
    simulator = uc.SequentialGaussianSimulator(4, covariance)

    result = simulator.simulate(n_realizations=1, seed=11, workers=8)

    assert result.workers == 1


@pytest.mark.parametrize(
    ('name', 'expected'),
    [
        ('simple', 'SimpleKriging'),
        ('SimpleKriging', 'SimpleKriging'),
        ('ordinary', 'OrdinaryKriging'),
        ('ordinary_kriging', 'OrdinaryKriging'),
    ],
)
def test_kriging_names_are_normalized(covariance, name, expected):
    simulator = uc.SequentialGaussianSimulator(4, covariance, kriging=name)

    assert simulator.kriging == expected


def test_configured_kriging_object_is_supported(covariance):
    kriging = uc.SimpleKriging(covariance, 4, mean=0.25)
    simulator = uc.SequentialGaussianSimulator(
        4,
        covariance,
        kriging=kriging,
        mean=0.25,
    )

    result = simulator.simulate(1, seed=3)

    assert result.kriging == 'SimpleKriging'


def test_configured_kriging_object_must_match_facade(covariance):
    other_covariance = uc.Gaussian(4, 1, 3, 1)

    with pytest.raises(ValueError, match='covariance'):
        uc.SequentialGaussianSimulator(
            4,
            covariance,
            kriging=uc.SimpleKriging(other_covariance, 4),
        )

    with pytest.raises(ValueError, match='grid_size'):
        uc.SequentialGaussianSimulator(
            4,
            covariance,
            kriging=uc.SimpleKriging(covariance, 5),
        )


def test_two_dimensional_grid_is_normalized(covariance):
    simulator = uc.SequentialGaussianSimulator((3, 2), covariance)

    result = simulator.simulate(1, seed=8)

    assert simulator.grid_size == (3, 2)
    assert result.shape == (1, 3, 2)


@pytest.mark.parametrize(
    ('kwargs', 'error'),
    [
        ({'grid_size': 0}, ValueError),
        ({'grid_size': (2,)}, ValueError),
        ({'grid_size': (2, -1)}, ValueError),
        ({'backend': 'gpu'}, ValueError),
        ({'kriging': 'universal'}, ValueError),
        ({'max_neighbors': -1}, ValueError),
        ({'iteration_limit': 0}, ValueError),
        ({'mean': True}, TypeError),
        ({'constant_path': 1}, TypeError),
        ({'covariance_cache': True}, ValueError),
        ({'min_value': 2, 'max_value': 2}, ValueError),
    ],
)
def test_configuration_validation(covariance, kwargs, error):
    defaults = {'grid_size': 4, 'covariance': covariance}

    with pytest.raises(error):
        uc.SequentialGaussianSimulator(**{**defaults, **kwargs})


@pytest.mark.parametrize(
    ('kwargs', 'error'),
    [
        ({'n_realizations': 0}, ValueError),
        ({'n_realizations': 1, 'workers': 0}, ValueError),
        ({'n_realizations': 1, 'seed': -1}, ValueError),
        ({'n_realizations': 1, 'seed': 2**31}, ValueError),
        ({'n_realizations': 1, 'seed': 1.5}, TypeError),
    ],
)
def test_execution_validation(covariance, kwargs, error):
    simulator = uc.SequentialGaussianSimulator(4, covariance)

    with pytest.raises(error):
        simulator.simulate(**kwargs)


def test_root_package_exports_complete_public_model_set():
    assert uc.Exponential is not None
    assert uc.OrdinaryKriging is not None
    assert uc.SequentialGaussianSimulator is not None
    assert uc.SimulationResult is not None


def test_c_backend_rejects_unsupported_configuration(covariance):
    with pytest.raises(ValueError, match='1D'):
        uc.SequentialGaussianSimulator((3, 2), covariance, backend='c')

    with pytest.raises(ValueError, match='zero mean'):
        uc.SequentialGaussianSimulator(3, covariance, backend='c', mean=1)


@pytest.mark.parametrize('model_class', [uc.Gaussian, uc.Exponential, uc.Spherical])
@pytest.mark.parametrize('kriging', ['simple', 'ordinary'])
def test_c_backend_supports_public_1d_models_and_kriging(model_class, kriging):
    simulator = uc.SequentialGaussianSimulator(
        8,
        model_class(8, 1, 4, sill=1.7, nugget=0.2),
        backend='c',
        kriging=kriging,
    )

    result = simulator.simulate(2, seed=17)

    assert result.shape == (2, 8, 1)
    assert np.isfinite(result.values).all()


def test_c_backend_result_has_the_same_shape_contract(covariance):
    simulator = uc.SequentialGaussianSimulator(4, covariance, backend='c')

    result = simulator.simulate(2, seed=7)

    assert result.shape == (2, 4, 1)
    assert result.backend == 'c'
