import pytest

import uc_sgsim as uc
from uc_sgsim.kriging import SimpleKriging
from uc_sgsim.cov_model import Gaussian, Spherical, Exponential
from uc_sgsim.exception import VariogramDoesNotCompute, IterationError
import os

X = 151
nR = 10
hs = 35
a = 17.32
bw = 1
C0 = 1
gaussian = Gaussian(hs, bw, a, C0)
spherical = Spherical(hs, bw, a, C0)
exponential = Exponential(hs, bw, a, C0)


@pytest.mark.sgsim
class TestUCSgsim:
    def sgsim_plot(self, sgsim: uc.UCSgsim) -> None:
        sgsim.plot()
        sgsim.plot([0, 1, 2])
        sgsim.save_plot()
        assert os.path.isfile('figure.png') is True
        sgsim.save_plot('test.jpg')
        assert os.path.isfile('test.jpg') is True
        sgsim.cdf_plot(x_location=10)
        sgsim.hist_plot(x_location=10)
        sgsim.get_variogram(n_processes=1)
        sgsim.variogram_plot()
        sgsim.mean_plot()
        sgsim.variance_plot()

    def sgsim_save(self, sgsim: uc.UCSgsim) -> None:
        sgsim.save_random_field('randomfield.csv', save_single=True)
        sgsim.save_variogram('variogram.csv', save_single=True)
        sgsim.save_random_field('randomfield/', save_single=False)
        sgsim.save_variogram('variogram/', save_single=False)

    @pytest.mark.parametrize(
        'kriging, cov_model, n_process, kwargs',
        [
            ('SimpleKriging', gaussian, 1, {}),
            ('SimpleKriging', exponential, 1, {}),
            ('SimpleKriging', spherical, 1, {}),
            ('SimpleKriging', gaussian, 1, {'max_value': 3, 'min_value': -3, 'max_neighbor': 6}),
            ('SimpleKriging', exponential, 1, {'max_value': 3, 'min_value': -3, 'max_neighbor': 6}),
            ('SimpleKriging', spherical, 1, {'max_value': 3, 'min_value': -3, 'max_neighbor': 6}),
            (
                'SimpleKriging',
                gaussian,
                1,
                {
                    'max_value': 3,
                    'min_value': -3,
                    'max_neighbor': 12,
                    'constant_path': True,
                    'cov_cache': True,
                },
            ),
            (
                'SimpleKriging',
                exponential,
                1,
                {
                    'max_value': 3,
                    'min_value': -3,
                    'max_neighbor': 12,
                    'constant_path': True,
                    'cov_cache': True,
                },
            ),
            (
                'SimpleKriging',
                spherical,
                1,
                {
                    'max_value': 3,
                    'min_value': -3,
                    'max_neighbor': 12,
                    'constant_path': True,
                    'cov_cache': True,
                },
            ),
            ('SimpleKriging', gaussian, 2, {}),
            ('SimpleKriging', exponential, 2, {}),
            ('SimpleKriging', spherical, 2, {}),
            ('SimpleKriging', gaussian, 2, {'max_value': 3, 'min_value': -3, 'max_neighbor': 6}),
            ('SimpleKriging', exponential, 2, {'max_value': 3, 'min_value': -3, 'max_neighbor': 6}),
            ('SimpleKriging', spherical, 2, {'max_value': 3, 'min_value': -3, 'max_neighbor': 6}),
            (
                'SimpleKriging',
                gaussian,
                2,
                {
                    'max_value': 3,
                    'min_value': -3,
                    'max_neighbor': 12,
                    'constant_path': True,
                    'cov_cache': True,
                },
            ),
            (
                'SimpleKriging',
                spherical,
                2,
                {
                    'max_value': 3,
                    'min_value': -3,
                    'max_neighbor': 12,
                    'constant_path': True,
                    'cov_cache': True,
                },
            ),
            (
                'SimpleKriging',
                exponential,
                2,
                {
                    'max_value': 3,
                    'min_value': -3,
                    'max_neighbor': 12,
                    'constant_path': True,
                    'cov_cache': True,
                },
            ),
            ('OrdinaryKriging', gaussian, 1, {}),
            ('OrdinaryKriging', exponential, 1, {}),
            ('OrdinaryKriging', spherical, 1, {}),
            ('OrdinaryKriging', gaussian, 1, {'max_value': 4, 'min_value': -4, 'max_neighbor': 6}),
            (
                'OrdinaryKriging',
                exponential,
                1,
                {'max_value': 4, 'min_value': -4, 'max_neighbor': 6},
            ),
            ('OrdinaryKriging', spherical, 1, {'max_value': 4, 'min_value': -4, 'max_neighbor': 6}),
            (
                'OrdinaryKriging',
                gaussian,
                1,
                {
                    'max_value': 3,
                    'min_value': -3,
                    'max_neighbor': 12,
                    'constant_path': True,
                    'cov_cache': True,
                },
            ),
            (
                'OrdinaryKriging',
                exponential,
                1,
                {
                    'max_value': 4,
                    'min_value': -4,
                    'max_neighbor': 12,
                    'constant_path': True,
                    'cov_cache': True,
                },
            ),
            (
                'OrdinaryKriging',
                spherical,
                1,
                {
                    'max_value': 3,
                    'min_value': -3,
                    'max_neighbor': 12,
                    'constant_path': True,
                    'cov_cache': True,
                },
            ),
            ('OrdinaryKriging', gaussian, 2, {}),
            ('OrdinaryKriging', exponential, 2, {}),
            ('OrdinaryKriging', spherical, 2, {}),
            ('OrdinaryKriging', gaussian, 2, {'max_value': 3, 'min_value': -3, 'max_neighbor': 6}),
            (
                'OrdinaryKriging',
                exponential,
                2,
                {'max_value': 4, 'min_value': -4, 'max_neighbor': 6},
            ),
            ('OrdinaryKriging', spherical, 2, {'max_value': 3, 'min_value': -3, 'max_neighbor': 6}),
            (
                'OrdinaryKriging',
                gaussian,
                2,
                {
                    'max_value': 4,
                    'min_value': -4,
                    'max_neighbor': 8,
                    'constant_path': True,
                    'cov_cache': True,
                },
            ),
            (
                'OrdinaryKriging',
                exponential,
                2,
                {
                    'max_value': 4,
                    'min_value': -4,
                    'max_neighbor': 8,
                    'constant_path': True,
                    'cov_cache': True,
                },
            ),
            (
                'OrdinaryKriging',
                spherical,
                2,
                {
                    'max_value': 4,
                    'min_value': -4,
                    'max_neighbor': 8,
                    'constant_path': True,
                    'cov_cache': True,
                },
            ),
        ],
    )
    def test_uc_sgsim_gaussian_py(self, kriging, cov_model, n_process, kwargs):
        sgsim = uc.UCSgsim(X, nR, cov_model, kriging=kriging, **kwargs)
        sgsim.run(n_processes=n_process, randomseed=454)
        sgsim.get_variogram(n_processes=n_process)
        self.sgsim_plot(sgsim)
        self.sgsim_save(sgsim)

    @pytest.mark.parametrize(
        'kriging, cov_model, n_process, kwargs',
        [
            ('SimpleKriging', gaussian, 1, {}),
            ('SimpleKriging', exponential, 1, {}),
            ('SimpleKriging', spherical, 1, {}),
            ('SimpleKriging', gaussian, 1, {'max_value': 3, 'min_value': -3, 'max_neighbor': 6}),
            ('SimpleKriging', exponential, 1, {'max_value': 3, 'min_value': -3, 'max_neighbor': 6}),
            ('SimpleKriging', spherical, 1, {'max_value': 3, 'min_value': -3, 'max_neighbor': 6}),
            (
                'SimpleKriging',
                gaussian,
                1,
                {
                    'max_value': 3,
                    'min_value': -3,
                    'max_neighbor': 12,
                    'constant_path': True,
                    'cov_cache': True,
                },
            ),
            (
                'SimpleKriging',
                exponential,
                1,
                {
                    'max_value': 3,
                    'min_value': -3,
                    'max_neighbor': 12,
                    'constant_path': True,
                    'cov_cache': True,
                },
            ),
            (
                'SimpleKriging',
                spherical,
                1,
                {
                    'max_value': 3,
                    'min_value': -3,
                    'max_neighbor': 12,
                    'constant_path': True,
                    'cov_cache': True,
                },
            ),
            ('SimpleKriging', gaussian, 2, {}),
            ('SimpleKriging', exponential, 2, {}),
            ('SimpleKriging', spherical, 2, {}),
            ('SimpleKriging', gaussian, 2, {'max_value': 3, 'min_value': -3, 'max_neighbor': 6}),
            ('SimpleKriging', exponential, 2, {'max_value': 3, 'min_value': -3, 'max_neighbor': 6}),
            ('SimpleKriging', spherical, 2, {'max_value': 3, 'min_value': -3, 'max_neighbor': 6}),
            (
                'SimpleKriging',
                gaussian,
                2,
                {
                    'max_value': 3,
                    'min_value': -3,
                    'max_neighbor': 12,
                    'constant_path': True,
                    'cov_cache': True,
                },
            ),
            (
                'SimpleKriging',
                spherical,
                2,
                {
                    'max_value': 3,
                    'min_value': -3,
                    'max_neighbor': 12,
                    'constant_path': True,
                    'cov_cache': True,
                },
            ),
            (
                'SimpleKriging',
                exponential,
                2,
                {
                    'max_value': 3,
                    'min_value': -3,
                    'max_neighbor': 12,
                    'constant_path': True,
                    'cov_cache': True,
                },
            ),
            ('OrdinaryKriging', gaussian, 1, {}),
            ('OrdinaryKriging', exponential, 1, {}),
            ('OrdinaryKriging', spherical, 1, {}),
            ('OrdinaryKriging', gaussian, 1, {'max_value': 6, 'min_value': -6, 'max_neighbor': 6}),
            (
                'OrdinaryKriging',
                exponential,
                1,
                {'max_value': 6, 'min_value': -6, 'max_neighbor': 6},
            ),
            ('OrdinaryKriging', spherical, 1, {'max_value': 6, 'min_value': -6, 'max_neighbor': 6}),
            (
                'OrdinaryKriging',
                gaussian,
                1,
                {
                    'max_value': 6,
                    'min_value': -6,
                    'max_neighbor': 12,
                    'constant_path': True,
                    'cov_cache': True,
                },
            ),
            (
                'OrdinaryKriging',
                exponential,
                1,
                {
                    'max_value': 6,
                    'min_value': -6,
                    'max_neighbor': 12,
                    'constant_path': True,
                    'cov_cache': True,
                },
            ),
            (
                'OrdinaryKriging',
                spherical,
                1,
                {
                    'max_value': 6,
                    'min_value': -6,
                    'max_neighbor': 12,
                    'constant_path': True,
                    'cov_cache': True,
                },
            ),
            ('OrdinaryKriging', gaussian, 2, {}),
            ('OrdinaryKriging', exponential, 2, {}),
            ('OrdinaryKriging', spherical, 2, {}),
            ('OrdinaryKriging', gaussian, 2, {'max_value': 6, 'min_value': -6, 'max_neighbor': 6}),
            (
                'OrdinaryKriging',
                exponential,
                2,
                {'max_value': 6, 'min_value': -6, 'max_neighbor': 6},
            ),
            ('OrdinaryKriging', spherical, 2, {'max_value': 6, 'min_value': -6, 'max_neighbor': 6}),
            (
                'OrdinaryKriging',
                gaussian,
                2,
                {
                    'max_value': 6,
                    'min_value': -6,
                    'max_neighbor': 8,
                    'constant_path': True,
                    'cov_cache': True,
                },
            ),
            (
                'OrdinaryKriging',
                exponential,
                2,
                {
                    'max_value': 6,
                    'min_value': -6,
                    'max_neighbor': 8,
                    'constant_path': True,
                    'cov_cache': True,
                },
            ),
            (
                'OrdinaryKriging',
                spherical,
                2,
                {
                    'max_value': 6,
                    'min_value': -6,
                    'max_neighbor': 8,
                    'constant_path': True,
                    'cov_cache': True,
                },
            ),
        ],
    )
    def test_uc_sgsim_gaussian_c(self, kriging, cov_model, n_process, kwargs):
        sgsim = uc.UCSgsim(X, nR, model=cov_model, kriging=kriging, engine='c', **kwargs)
        sgsim.run(n_processes=n_process, randomseed=454)
        sgsim.get_variogram(n_processes=n_process)
        self.sgsim_plot(sgsim)
        self.sgsim_save(sgsim)

    def test_uc_wrong_kriging_method(self):
        with pytest.raises(TypeError):
            uc.UCSgsim(  # noqa
                X,
                nR,
                gaussian,
                kriging='wrongKriging',
            )

    def test_uc_sgsim_variogram_exception(self):
        sgsim = uc.UCSgsim(X, nR, gaussian, engine='c')
        sgsim.run(n_processes=1, randomseed=454)
        with pytest.raises(VariogramDoesNotCompute):
            sgsim.variogram_plot()

        sgsim = uc.UCSgsim(X, nR, gaussian)
        sgsim.run(n_processes=1, randomseed=454)
        with pytest.raises(VariogramDoesNotCompute):
            sgsim.variogram_plot()
        with pytest.raises(VariogramDoesNotCompute):
            sgsim.save_variogram('variogram/', save_single=False)

    def test_sgsim_plot_params(self):
        import numpy as np

        sgsim = uc.UCSgsim(X, nR, gaussian)
        vbandwidth = np.arange(0, hs, bw)
        np.testing.assert_allclose(sgsim.vbandwidth, vbandwidth)
        assert sgsim.vmodel == gaussian
        assert sgsim.vmodel_name == 'Gaussian'
        assert sgsim.vbandwidth_step == 1
        assert sgsim.vsill == 1
        assert sgsim.vk_range == 17.32

    def test_get_all_attributes(self):
        sgsim = uc.UCSgsim(
            X,
            nR,
            gaussian,
            max_value=3,
            min_value=-3,
            max_neighbor=12,
            constant_path=True,
            cov_cache=True,
        )
        attributes = sgsim.get_all_attributes()
        assert attributes['grid_size'] == 151
        assert attributes['realization_number'] == 10
        assert attributes['max_value'] == 3
        assert attributes['min_value'] == -3
        assert attributes['max_neighbor'] == 12
        assert attributes['constant_path'] is True
        assert attributes['cov_cache'] is True
        assert isinstance(attributes['kriging'], SimpleKriging)
        assert attributes['model'] == gaussian

    def test_sgsim_repr(self, capsys):
        sgsim = uc.UCSgsim(X, nR, gaussian, engine='c')
        # Call repr() to get the string representation
        repr_output = repr(sgsim)
        # Print the repr output (optional)
        print(repr_output)

        # Capture the printed output
        captured = capsys.readouterr()

        # Assert the printed output
        assert (
            captured.out.strip()
            == 'SgsimField(151, 10, SimpleKriging, GaussianModel'
            + '(bandwidth_len=35, bandwidth_step=1, k_range=17.32, sill=1, nugget=0))'
        )
        assert captured.err == ''
        assert (
            repr_output
            == 'SgsimField(151, 10, SimpleKriging, GaussianModel'
            + '(bandwidth_len=35, bandwidth_step=1, k_range=17.32, sill=1, nugget=0))'
        )

    def test_sgsim_reach_iteration_limit(self):
        sgsim = uc.UCSgsim(
            X,
            100,
            gaussian,
            max_value=1,
            min_value=-1,
            iteration_limit=3,
            max_neighbor=12,
        )
        with pytest.raises(IterationError):
            sgsim.run(n_processes=1, randomseed=12312)

    def test_sgsim_with_wrong_setting(self):
        with pytest.raises(ValueError):
            uc.UCSgsim(
                X,
                100,
                gaussian,
                max_value=1,
                min_value=-1,
                iteration_limit=3,
                max_neighbor=12,
                constant_path=False,
                cov_cache=True,
            )

        with pytest.raises(TypeError):
            uc.UCSgsim(
                X,
                100,
                gaussian,
                kriging=1234,
                max_value=1,
                min_value=-1,
                iteration_limit=3,
                max_neighbor=12,
            )

    def test_get_attributes(self):
        sgsim = uc.UCSgsim(
            X,
            100,
            gaussian,
            max_value=1,
            min_value=-1,
            iteration_limit=3,
            max_neighbor=12,
        )
        assert sgsim.constant_path is False
        assert sgsim.cov_cache is False
        print(sgsim.kriging)
        print(sgsim.bandwidth_step)
        assert sgsim.max_neighbor == 12
        assert sgsim.min_value == -1
        assert sgsim.max_value == 1
        assert len(sgsim.grid) == 151
        assert len(sgsim.grid[0]) == 1
        assert sgsim.y_size == 1
