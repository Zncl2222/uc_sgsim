import pytest

import numpy as np
from uc_sgsim.kriging import SimpleKriging, OrdinaryKriging
from uc_sgsim.cov_model import Gaussian


@pytest.mark.kriging
class TestSimpleKriging:
    @classmethod
    def setup_class(cls) -> None:
        cls.bw = 1
        cls.hs = 35
        cls.a = 17.32
        cls.C0 = 1
        cls.x_len = 150
        cls.x_grid = np.linspace(0, cls.x_len, cls.x_len + 1).reshape(cls.x_len + 1, 1)
        cls.gaussian = Gaussian(cls.hs, cls.bw, cls.a, cls.C0)
        cls.kriging = SimpleKriging(cls.gaussian, cls.x_len, cov_cache=False)
        cls.o_kriging = OrdinaryKriging(cls.gaussian, cls.x_len, cov_cache=False)
        cls.sampled = np.array(
            [
                (0, 0),
                (10, 0),
                (10, 20),
                (10, 74),
                (10, 85),
                (10, 115),
                (10, 146),
            ],
        )

    def test_simple_kriging_prediction(self):
        res = np.empty(self.x_len)
        for i in range(len(self.sampled[0])):
            res[int(self.sampled.T[0][i])] = self.sampled.T[1][i]

        for i in [(x, 0) for x in range(self.x_len)]:
            if i not in self.sampled[0]:
                res[i], kriging_std = self.kriging.prediction(i, self.sampled)
                assert kriging_std >= 0

    def test_simple_kriging_simulation(self):
        np.random.seed(123456)
        x = np.array([0, 150]).reshape(2, 1)
        y_value = np.zeros(2).reshape(2, 1)
        mesh = np.hstack([x, y_value])
        unsampled = np.linspace(0, 150, 151)
        unsampled = np.delete(unsampled, [0, -1])

        neighbor = 0
        estimation = self.kriging.simulation(75, mesh, neighbor=neighbor)
        assert isinstance(estimation, float)
        assert pytest.approx(estimation, 1e-7) == 0.4691123

    def test_ordinary_kriging_prediction(self):
        res = np.empty(self.x_len)
        for i in range(len(self.sampled[0])):
            res[int(self.sampled.T[0][i])] = self.sampled.T[1][i]

        for i in [(x, 0) for x in range(self.x_len)]:
            if i not in self.sampled[0]:
                res[i], kriging_std = self.kriging.prediction(i, self.sampled)
                assert kriging_std >= 0

    def test_ordinary_kriging_simulation(self):
        np.random.seed(123456)
        x = np.array([0, 150]).reshape(2, 1)
        y_value = np.zeros(2).reshape(2, 1)
        mesh = np.hstack([x, y_value])
        unsampled = np.linspace(0, 150, 151)
        unsampled = np.delete(unsampled, [0, -1])

        neighbor = 0
        estimation = self.o_kriging.simulation(75, mesh, neighbor=neighbor)
        assert isinstance(estimation, float)
        assert pytest.approx(estimation, 1e-7) == 0.4691123

    def test_kriging_repr(self, capsys):
        # Call repr() to get the string representation
        repr_output = repr(self.kriging)
        # Print the repr output (optional)
        print(repr_output)

        # Capture the printed output
        captured = capsys.readouterr()

        # Assert the printed output
        assert captured.out.strip() == 'SimpleKriging'
        assert captured.err == ''
        assert repr_output == 'SimpleKriging'
