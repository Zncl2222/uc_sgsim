"""High-level, result-oriented API for sequential Gaussian simulation."""

from __future__ import annotations

import copy
import secrets
from dataclasses import dataclass
from typing import Literal, Optional, Sequence, Tuple, Union

import numpy as np

from uc_sgsim.cov_model.base import CovModel
from uc_sgsim.cov_model.model import Gaussian
from uc_sgsim.kriging import Kriging, OrdinaryKriging, SimpleKriging
from uc_sgsim.sgsim import UCSgsim

Backend = Literal['python', 'c']
GridSize = Union[int, Sequence[int]]
KrigingConfiguration = Union[str, Kriging]

__all__ = ['SequentialGaussianSimulator', 'SimulationResult']


@dataclass(frozen=True)
class SimulationResult:
    """Values and execution metadata produced by a simulation.

    The stored array is copied and marked read-only. Use :meth:`to_numpy`
    when a mutable copy is needed.
    """

    values: np.ndarray
    seed: int
    workers: int
    backend: Backend
    covariance: CovModel
    kriging: str

    def __post_init__(self) -> None:
        if not isinstance(self.covariance, CovModel):
            raise TypeError('covariance must be a CovModel instance')
        if self.backend not in ('python', 'c'):
            raise ValueError("backend must be either 'python' or 'c'")
        object.__setattr__(self, 'seed', _seed(self.seed))
        object.__setattr__(self, 'workers', _positive_int(self.workers, 'workers'))

        values = np.array(self.values, dtype=float, copy=True)
        if values.ndim != 3:
            raise ValueError('simulation values must have shape (realizations, x, y)')
        if values.shape[0] == 0:
            raise ValueError('simulation values must contain at least one realization')
        values.setflags(write=False)
        object.__setattr__(self, 'values', values)

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Shape of the values as ``(realizations, x, y)``."""
        return self.values.shape

    @property
    def n_realizations(self) -> int:
        """Number of generated realizations."""
        return self.values.shape[0]

    @property
    def grid_shape(self) -> Tuple[int, int]:
        """Normalized two-dimensional grid shape."""
        return self.values.shape[1], self.values.shape[2]

    def to_numpy(self, *, copy: bool = True) -> np.ndarray:
        """Return the values, copying by default to preserve result integrity."""
        return self.values.copy() if copy else self.values

    def __array__(
        self,
        dtype: Optional[np.dtype] = None,
        copy: Optional[bool] = None,
    ) -> np.ndarray:
        values = np.asarray(self.values, dtype=dtype)
        return values.copy() if copy is True else values


class SequentialGaussianSimulator:
    """Configure a simulator once and return a result from each execution.

    This class is a high-level facade over :class:`UCSgsim`. It deliberately
    keeps the validated scientific implementation unchanged while presenting
    clearer configuration names and stable realization-count semantics.

    Args:
        grid_size: A positive 1D size or a two-item ``(x, y)`` size.
        covariance: Covariance model used by kriging.
        kriging: ``"simple"``, ``"ordinary"``, their legacy class names,
            or a configured :class:`Kriging` instance.
        backend: Simulation backend, either ``"python"`` or ``"c"``.
        mean: Known global mean used by simple kriging.
        max_neighbors: Maximum number of previously sampled neighbors.
        constant_path: Reuse a random path after the first realization.
        covariance_cache: Cache covariance values. This requires
            ``constant_path=True``.
        iteration_limit: Maximum rejected realizations before failing.
        min_value: Optional lower whole-realization rejection bound.
        max_value: Optional upper whole-realization rejection bound.
    """

    def __init__(
        self,
        grid_size: GridSize,
        covariance: CovModel,
        *,
        kriging: KrigingConfiguration = 'simple',
        backend: Backend = 'python',
        mean: float = 0.0,
        max_neighbors: int = 8,
        constant_path: bool = False,
        covariance_cache: bool = False,
        iteration_limit: int = 10,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ) -> None:
        self._grid_size = _normalize_grid_size(grid_size)
        if not isinstance(covariance, CovModel):
            raise TypeError('covariance must be a CovModel instance')
        if backend not in ('python', 'c'):
            raise ValueError("backend must be either 'python' or 'c'")
        if backend == 'c' and not isinstance(self._grid_size, int):
            raise ValueError('the c backend currently supports only 1D grids')
        if backend == 'c' and not isinstance(covariance, Gaussian):
            raise ValueError('the c backend currently supports only Gaussian covariance')

        normalized_kriging, kriging_name = _normalize_kriging(kriging)
        if backend == 'c' and kriging_name != 'SimpleKriging':
            raise ValueError('the c backend currently supports only simple kriging')

        self._covariance = covariance
        self._kriging = normalized_kriging
        self._kriging_name = kriging_name
        self._backend = backend
        self._mean = _finite_float(mean, 'mean')
        if self._backend == 'c' and self._mean != 0.0:
            raise ValueError('the c backend currently supports only a zero mean')
        self._max_neighbors = _non_negative_int(max_neighbors, 'max_neighbors')
        self._constant_path = _boolean(constant_path, 'constant_path')
        self._covariance_cache = _boolean(covariance_cache, 'covariance_cache')
        if self._covariance_cache and not self._constant_path:
            raise ValueError('covariance_cache requires constant_path=True')
        self._iteration_limit = _positive_int(iteration_limit, 'iteration_limit')
        self._min_value = _optional_finite_float(min_value, 'min_value')
        self._max_value = _optional_finite_float(max_value, 'max_value')
        if (
            self._min_value is not None
            and self._max_value is not None
            and self._min_value >= self._max_value
        ):
            raise ValueError('min_value must be smaller than max_value')
        if isinstance(self._kriging, Kriging):
            _validate_configured_kriging(
                self._kriging,
                covariance=self._covariance,
                grid_size=self._grid_size,
                mean=self._mean,
                covariance_cache=self._covariance_cache,
            )
        self._last_result: Optional[SimulationResult] = None

    @property
    def grid_size(self) -> Union[int, Tuple[int, int]]:
        return self._grid_size

    @property
    def covariance(self) -> CovModel:
        return self._covariance

    @property
    def kriging(self) -> str:
        return self._kriging_name

    @property
    def backend(self) -> Backend:
        return self._backend

    @property
    def last_result(self) -> Optional[SimulationResult]:
        """Most recent successful result, or ``None`` before the first run."""
        return self._last_result

    def simulate(
        self,
        n_realizations: int,
        *,
        seed: Optional[int] = None,
        workers: int = 1,
    ) -> SimulationResult:
        """Generate exactly ``n_realizations`` and return an immutable result.

        The legacy engine assigns the same-sized batch to every worker. When
        the requested total is not divisible by ``workers``, this facade runs
        one small extra batch and trims it from the returned result.
        """
        requested_count = _positive_int(n_realizations, 'n_realizations')
        worker_count = min(_positive_int(workers, 'workers'), requested_count)
        resolved_seed = _seed(seed)
        batch_size = (requested_count + worker_count - 1) // worker_count

        simulator = UCSgsim(
            grid_size=_legacy_grid_size(self._grid_size),
            realization_number=batch_size,
            model=self._covariance,
            kriging=_copy_kriging(self._kriging),
            engine=self._backend,
            mean=self._mean,
            max_neighbor=self._max_neighbors,
            constant_path=self._constant_path,
            cov_cache=self._covariance_cache,
            iteration_limit=self._iteration_limit,
            min_value=self._min_value,
            max_value=self._max_value,
        )
        simulator.run(n_processes=worker_count, randomseed=resolved_seed)

        values = np.asarray(simulator.random_fields)
        if values.ndim == 2:
            values = values[:, :, np.newaxis]

        result = SimulationResult(
            values=values[:requested_count],
            seed=resolved_seed,
            workers=worker_count,
            backend=self._backend,
            covariance=self._covariance,
            kriging=self._kriging_name,
        )
        self._last_result = result
        return result


def _normalize_grid_size(grid_size: GridSize) -> Union[int, Tuple[int, int]]:
    if isinstance(grid_size, (int, np.integer)) and not isinstance(grid_size, bool):
        return _positive_int(grid_size, 'grid_size')
    if isinstance(grid_size, (str, bytes)):
        raise TypeError('grid_size must be an integer or a two-item sequence')
    try:
        dimensions = tuple(grid_size)
    except TypeError as error:
        raise TypeError(
            'grid_size must be an integer or a two-item sequence',
        ) from error
    if len(dimensions) != 2:
        raise ValueError('grid_size must contain exactly two dimensions')
    return (
        _positive_int(dimensions[0], 'grid_size[0]'),
        _positive_int(dimensions[1], 'grid_size[1]'),
    )


def _normalize_kriging(
    kriging: KrigingConfiguration,
) -> Tuple[KrigingConfiguration, str]:
    if isinstance(kriging, str):
        normalized = kriging.lower().replace('_', '').replace('-', '').replace(' ', '')
        names = {
            'simple': 'SimpleKriging',
            'simplekriging': 'SimpleKriging',
            'ordinary': 'OrdinaryKriging',
            'ordinarykriging': 'OrdinaryKriging',
        }
        try:
            name = names[normalized]
        except KeyError as error:
            raise ValueError("kriging must be either 'simple' or 'ordinary'") from error
        return name, name
    if not isinstance(kriging, (SimpleKriging, OrdinaryKriging)):
        raise TypeError(
            'kriging must be a name or a SimpleKriging/OrdinaryKriging instance',
        )
    return kriging, type(kriging).__name__


def _legacy_grid_size(grid_size: Union[int, Tuple[int, int]]) -> Union[int, list[int]]:
    return grid_size if isinstance(grid_size, int) else list(grid_size)


def _copy_kriging(kriging: KrigingConfiguration) -> KrigingConfiguration:
    return kriging if isinstance(kriging, str) else copy.deepcopy(kriging)


def _validate_configured_kriging(
    kriging: Kriging,
    *,
    covariance: CovModel,
    grid_size: Union[int, Tuple[int, int]],
    mean: float,
    covariance_cache: bool,
) -> None:
    if kriging.model is not covariance:
        raise ValueError('the Kriging object must use the configured covariance')

    expected_grid = (grid_size, 0) if isinstance(grid_size, int) else grid_size
    if (kriging.x_size, kriging.y_size) != expected_grid:
        raise ValueError('the Kriging object must use the configured grid_size')
    if not np.isclose(kriging.mean, mean):
        raise ValueError('the Kriging object must use the configured mean')

    kriging_uses_cache = hasattr(kriging, '_cov_cache')
    if kriging_uses_cache != covariance_cache:
        raise ValueError('the Kriging object must use the configured covariance_cache')


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _non_negative_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _finite_float(value: float, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a finite number") from error
    if not np.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def _optional_finite_float(value: Optional[float], name: str) -> Optional[float]:
    return None if value is None else _finite_float(value, name)


def _boolean(value: bool, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a boolean")
    return bool(value)


def _seed(seed: Optional[int]) -> int:
    if seed is None:
        return secrets.randbelow(2**31)
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError('seed must be an integer or None')
    if seed < 0 or seed >= 2**31:
        raise ValueError('seed must satisfy 0 <= seed < 2**31')
    return int(seed)
