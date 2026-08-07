from __future__ import annotations

from typing import Optional

import numpy as np
from uc_sgsim.plotting import SgsimPlot
from uc_sgsim.exception import VariogramDoesNotCompute
from uc_sgsim.kriging import SimpleKriging, OrdinaryKriging, Kriging
from uc_sgsim.utils import save_as_multiple_file, save_as_one_file
from uc_sgsim.cov_model.base import CovModel


class RandomField:
    """
    The based class for the random field object.

    This class is a parent class for any kind of randomfiled.

    Methods:
        _create_grid: create the grid for the random field.
        save_random_field: save the random field data to files.
        save_variogram: save the variogram data to files.
    """

    def __init__(self, grid_size: int | list[int, int], n_realizations: int):
        """
        Initialize a random field object.

        Args:
            grid_size (int | list[int, int]): Size of the grid for the random field.
            n_realizations (int): Number of realizations.
        """
        self._n_realizations = n_realizations
        self._grid_size = grid_size
        self._create_grid(grid_size)

    def _create_grid(self, grid_size: int | list[int, int]) -> None:
        self._grid = (
            np.zeros((grid_size, 1))
            if isinstance(grid_size, int)
            else np.zeros((grid_size[0], grid_size[1]))
        )
        self._x_size = grid_size if isinstance(grid_size, int) else grid_size[0]
        self._y_size = 1 if isinstance(grid_size, int) else grid_size[1]
        self._coordinate = np.array(
            [np.array((x, y)) for x in range(self._x_size) for y in range(self._y_size)],
        )
        self.variogram = 0

    @property
    def grid_size(self) -> int | list[int, int]:
        return self._grid_size

    @property
    def grid(self) -> int:
        return self._grid

    @property
    def x_size(self) -> int:
        return self._x_size

    @property
    def y(self) -> int:
        return self._y

    @property
    def y_size(self) -> int:
        return self._y_size

    @property
    def n_realizations(self) -> int:
        return self._n_realizations

    @n_realizations.setter
    def realization_number(self, val: int):
        self._n_realizations = val

    def save_random_field(
        self,
        path: str,
        file_type: str = 'csv',
        save_single: bool = False,
    ) -> None:
        """
        Save random field data to files.

        Args:
            path (str): Path to the directory or file where data will be saved.
            file_type (str): File extension for saved files (default is 'csv').
            save_single (bool): Save as a single file or multiple files (default is False).

        Returns:
            None
        """
        digit = int(np.log10(self.realization_number))
        number_head = ''
        for i in range(digit):
            number_head += '0'
        num_val = 1
        if save_single is False:
            for i in range(self.realization_number):
                if i // num_val == 10:
                    num_val *= 10
                    number_head = number_head[:-1]
                number = number_head + str(i)
                save_as_multiple_file(
                    number,
                    self.x_size,
                    self.random_fields,
                    file_type,
                    'Realizations',
                )
        else:
            save_as_one_file(path, self.random_fields)

    def save_variogram(self, path: str, file_type: str = 'csv', save_single: bool = False) -> None:
        """
        Save variogram data to files.

        Args:
            path (str): Path to the directory or file where data will be saved.
            file_type (str): File extension for saved files (default is 'csv').
            save_single (bool): Save as a single file or multiple files (default is False).

        Returns:
            None

        Raises:
            VariogramDoesNotCompute: If variogram does not compute.
        """
        if type(self.variogram) == int:
            raise VariogramDoesNotCompute()
        digit = int(np.log10(self.realization_number))
        number_head = ''
        for i in range(digit):
            number_head += '0'
        num_val = 1
        if save_single is False:
            for i in range(self.realization_number):
                if i // num_val == 10:
                    num_val *= 10
                    number_head = number_head[:-1]
                number = number_head + str(i)
                save_as_multiple_file(
                    number,
                    len(self.bandwidth),
                    self.variogram,
                    file_type,
                    'Variogram',
                )
        else:
            save_as_one_file(path, self.variogram)


class SgsimField(RandomField, SgsimPlot):
    """
    SgsimField represents a random field for Sequential Gaussian Simulation (SGSIM).

    This class extends the RandomField and SgsimPlot classes to provide functionality
    for simulating random fields using the Sequential Gaussian Simulation method.

    Attributes:
        model (CovModel): The covariance model used for simulation.
        bandwidth (np.array): The bandwidth values used in the simulation.
        bandwidth_step (int): The bandwidth step used in the simulation.
        min_value (float): The minimum value for simulation.
        max_value (float): The maximum value for simulation.
        max_neighbor (int): The maximum number of neighbors considered in kriging.

    Methods:
        _set_kriging_method: Set the kriging method based on provided parameters.
        _set_kwargs: Set additional keyword arguments.
    """

    def __init__(
        self,
        grid_size: int | list[int, int],
        n_realizations: int,
        model: CovModel,
        kriging: str | Kriging = 'SimpleKriging',
        constant_path: bool = False,
        cov_cache: bool = False,
        max_neighbor: int = 8,
        iteration_limit: int = 10,
        max_value: Optional[float] = None,
        min_value: Optional[float] = None,
        mean: float = 0.0,
    ):
        """
        Initialize a Sgsim field.

        Args:
            grid_size (int | list[int, int]): Size of the grid for the field.
            n_realizations (int): Number of realizations.
            model (CovModel): Covariance model for simulation.
            kriging (str | Kriging): Kriging method to use (default is 'SimpleKriging').
            mean (float): Known global mean used by Simple Kriging.
            min_value (float, optional): Explicit lower rejection bound. The
                default is unbounded.
            max_value (float, optional): Explicit upper rejection bound. The
                default is unbounded.
        """
        RandomField.__init__(self, grid_size, n_realizations)
        SgsimPlot.__init__(self, model)

        self._model = model
        self._bandwidth_step = model.bandwidth_step
        self._bandwidth = model.bandwidth
        self._kriging = kriging
        self._constant_path = constant_path
        self._use_cov_cache = cov_cache
        if not isinstance(max_neighbor, (int, np.integer)) or max_neighbor < 0:
            raise ValueError('max_neighbor must be a non-negative integer')
        self._max_neighbor = int(max_neighbor)
        self._max_value = max_value
        self._min_value = min_value
        self._mean = float(mean)
        self.iteration_limit = iteration_limit
        self._set_kriging_method()
        self._set_default_value()

    @property
    def model(self) -> CovModel:
        return self._model

    @property
    def bandwidth(self) -> np.array:
        return self._bandwidth

    @property
    def bandwidth_step(self) -> int:
        return self._bandwidth_step

    @property
    def max_value(self) -> float:
        return self._max_value

    @property
    def min_value(self) -> float:
        return self._min_value

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def max_neighbor(self) -> int:
        return self._max_neighbor

    @property
    def constant_path(self) -> bool:
        return self._constant_path

    @property
    def cov_cache(self) -> bool:
        return self._use_cov_cache

    @property
    def kriging(self) -> Kriging:
        return self._kriging

    def _set_kriging_method(self) -> None:
        """
        Set the kriging method based on provided parameters.

        Args:
            kriging (str | Kriging): Kriging method to use.
            **kwargs: Additional keyword arguments.
            (The kwargs setting will be refactor in future version).

        Returns:
            None

        Raises:
            ValueError: If covcahe is True and constant_path is False.
            TypeError: If kriging arg is not acceptable.

        """
        if self._constant_path is False and self._use_cov_cache is True:
            raise ValueError('cov_cache should be False when constant_path is False')

        if self._kriging == 'SimpleKriging':
            self._kriging = SimpleKriging(
                self.model,
                self.grid_size,
                self._use_cov_cache,
                mean=self._mean,
            )
        elif self._kriging == 'OrdinaryKriging':
            self._kriging = OrdinaryKriging(
                self.model,
                self.grid_size,
                self._use_cov_cache,
                mean=self._mean,
            )
        else:
            if not isinstance(self._kriging, (SimpleKriging, OrdinaryKriging)):
                raise TypeError('Kriging should be class SimpleKriging or OrdinaryKriging')
            if not np.isclose(self._kriging.mean, self._mean):
                raise ValueError('mean must match the mean configured on the Kriging object')

    def _set_default_value(self) -> None:
        self._min_value = -np.inf if self._min_value is None else float(self._min_value)
        self._max_value = np.inf if self._max_value is None else float(self._max_value)
        if np.isnan(self._min_value) or np.isnan(self._max_value):
            raise ValueError('min_value and max_value must not be NaN')
        if self._min_value >= self._max_value:
            raise ValueError('min_value must be smaller than max_value')

    def get_all_attributes(self) -> dict:
        """
        Get all attributes of the SgsimField object.

        Returns:
            dict: All attributes of the SgsimField object.
        """
        return {
            'grid_size': self.grid_size,
            'realization_number': self.realization_number,
            'model': self.model,
            'kriging': self._kriging,
            'constant_path': self._constant_path,
            'cov_cache': self._use_cov_cache,
            'mean': self._mean,
            'min_value': self._min_value,
            'max_value': self._max_value,
            'max_neighbor': self._max_neighbor,
        }

    def __repr__(self) -> str:
        return (
            f'SgsimField({self.grid_size}, {self.realization_number}, '
            f'{self._kriging}, {self.model})'
        )
