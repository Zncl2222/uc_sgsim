from __future__ import annotations

import copy
import time
import sys
from pathlib import Path
from ctypes import CDLL, POINTER, c_double, c_int
from typing import Literal, Optional
from multiprocessing import Pool

import numpy as np
from uc_sgsim.random_field import SgsimField
from uc_sgsim.cov_model.base import CovModel
from uc_sgsim.kriging import Kriging
from uc_sgsim.utils import CovModelStructure, SgsimStructure
from .exception import IterationError

BASE_DIR = Path(__file__).resolve().parent


class UCSgsim(SgsimField):
    """
    A class to perform Unconditional Sequential Gaussian Simulation (SGSim)
    for random field generation.

    This class supports both Python and C engines to generate realizations
    and compute variograms
    for random fields, with parallel processing capabilities.

    Args:
        grid_size (int | list[int, int]): The size of the grid for the
            simulation. Can be a single integer for a 1D grid or a list
            of two integers for a 2D grid.
        realization_number (int): The number of realizations to generate.
        model (CovModel): The covariance model used for the simulation.
        kriging (str | Kriging, optional): The kriging method used for
            interpolation, either a string specifying the method
            ('SimpleKriging' or 'OrdinaryKriging') or a custom Kriging
            object. Defaults to 'SimpleKriging'.
        engine (Literal['python', 'c'], optional): The engine to run
            the simulation ('python' or 'c'). Defaults to 'python'.
        mean (float, optional): Known global mean for Simple Kriging.
            Defaults to 0.
        min_value (float, optional): Explicit lower whole-realization
            rejection bound. Defaults to no lower bound.
        max_value (float, optional): Explicit upper whole-realization
            rejection bound. Defaults to no upper bound.

    Methods:
        run(n_processes=1, randomseed=123):
            Run the simulation using the selected engine.

        get_variogram(n_processes=1):
            Compute the variogram of the generated random field using the
                selected engine.

        _run_python(n_process, randomseed) -> np.array:
            Generate random field realizations using Python multiprocessing.

        _get_variogram_python(n_process=1) -> None:
            Compute the variogram of the generated random field using
                Python multiprocessing.

        _simulation_python(randomseed=0, parallel=False) -> np.array:
            Perform the simulation process for a subset of realizations using Python.

        _read_shared_lib() -> CDLL:
            Load the shared library (.so or .dll) for the C-based simulation.

        _run_c(n_process, randomseed) -> np.array:
            Generate random field realizations using the C engine.

        _simulation_c(randomseed) -> np.array:
            Perform simulation using the shared library and return the
                generated realizations.

        _get_variogram_c(n_process=1) -> np.array:
            Compute the variogram of the generated random field using the C engine.

        _compute_variogram_c(n_process) -> np.array:
            Compute the variogram of a subset of realizations using the shared library.

        _reshape_simulaiton(process_results, n_process):
            Reshape the simulation results from parallel processes into a single array.
    """

    def __init__(
        self,
        grid_size: int | list[int, int],
        realization_number: int,
        model: CovModel,
        kriging: str | Kriging = 'SimpleKriging',
        engine: Literal['python', 'c'] = 'python',
        constant_path: bool = False,
        cov_cache: bool = False,
        max_neighbor: int = 8,
        iteration_limit: int = 10,
        max_value: Optional[float] = None,
        min_value: Optional[float] = None,
        mean: float = 0.0,
    ):
        super().__init__(
            grid_size=grid_size,
            n_realizations=realization_number,
            model=model,
            kriging=kriging,
            constant_path=constant_path,
            cov_cache=cov_cache,
            max_neighbor=max_neighbor,
            iteration_limit=iteration_limit,
            max_value=max_value,
            min_value=min_value,
            mean=mean,
        )
        self.engine = engine

    def run(self, n_processes: int = 1, randomseed: Optional[int] = None):
        """
        Run the Unconditional Sequential Gaussian Simulation (SGSim) using the specified engine.

        Depending on the selected engine ('python' or 'c'), this method will run the simulation
        either using Python's multiprocessing or the C engine for faster performance. The number
        of processes used for parallel computing can be adjusted.

        Args:
            n_processes (int, optional): Number of parallel processes to use for the simulation.
                                        Defaults to 1.
            randomseed (int, optional): Seed for random number generation to ensure reproducibility.
                                        Defaults to 123.

        Raises:
            ValueError: If the specified engine is not 'python' or 'c'.
        """
        if self.engine == 'python':
            self._run_python(n_process=n_processes, randomseed=randomseed)
        elif self.engine == 'c':
            self._run_c(n_process=n_processes, randomseed=randomseed)
        else:
            raise ValueError('Invalid engine')

    def get_variogram(self, n_processes: int = 1):
        if self.engine == 'python':
            self._get_variogram_python(n_process=n_processes)
        elif self.engine == 'c':
            self._get_variogram_c(n_process=n_processes)
        else:
            raise ValueError('Invalid engine')

    def _run_python(self, n_process: int, randomseed: int) -> np.array:
        """
        Generate random field realizations.

        Args:
            n_process (int): Number of parallel processes to use.
            randomseed (int): Seed for random number generation.

        Returns:
            np.array: Array containing generated random field realizations.
        """
        # Create pool and distribute realizations to each process.
        pool = Pool(processes=n_process)
        self.n_process = n_process
        self.realization_number = self.realization_number * n_process
        self.random_fields = np.empty([self.realization_number, self._x_size, self._y_size])

        # Prepare the args for each processes
        if randomseed is None:
            randomseed = np.random.randint(0, 2**31 - 1)
        rand_list = [randomseed + i for i in range(n_process)]

        parallel = [True] * n_process

        # Start parallel computing
        try:
            _simulation = pool.starmap(self._simulation_python, zip(rand_list, parallel))
        except IterationError:
            # We should handle the error like this to measure the coverage of sub process
            pool.close()
            pool.join()
            raise IterationError()

        pool.close()
        # Use pool.join() to measure the coverage of sub process
        pool.join()

        # Collect data from each process
        self._reshape_simulaiton(_simulation, n_process)

        return self.random_fields

    def _get_variogram_python(self, n_process: int = 1) -> None:
        """
        Compute the variogram of the generated random field.

        Args:
            n_process (int, optional): Number of parallel processes to use (default is 1).
        """
        # Create pool and args than distribute realizations and args to each process.
        pool = Pool(processes=n_process)
        model_len = self.x_size
        x = np.linspace(0, self.x_size - 1, model_len).reshape(model_len, 1)
        grid = [
            np.hstack([x, self.random_fields[i, :].reshape(model_len, 1)])
            for i in range(self.realization_number)
        ]

        # Start computing
        self.variogram = pool.starmap(self.model.variogram, zip(grid))
        pool.close()
        # Use pool.join() to measure the coverage of sub process
        pool.join()
        self.variogram = np.array(self.variogram)

    def _simulation_python(self, randomseed: int = 0, parallel: bool = False) -> np.array:
        """
        Perform the simulation process for a subset of realizations.

        Args:
            randomseed (int, optional): Seed for random number generation (default is 0).
            parallel (bool, optional): Flag to enable parallel processing (default is False).

        Returns:
            np.array: Array containing generated random field realizations.
        """
        if randomseed is not None:
            np.random.seed(randomseed)
        self.n_process = 1 if parallel is False else self.n_process
        counts = 0
        iteration_failed = 0
        start_time = time.time()

        unsampled = copy.deepcopy(self._coordinate)
        # Loop for randomfield generation
        while counts < (self.realization_number // self.n_process):
            # Grid and initial value preparing
            drop = False
            neighbor = 0

            # If simulate with constant_path then draw randompath only once
            if not self._constant_path or counts == 0:
                np.random.shuffle(unsampled)

            sampled = np.empty((0, 4), dtype=float)
            # Loop for kriging simulation
            for coordinate in unsampled:
                x = int(coordinate[0])
                y = int(coordinate[1])
                self._grid[x][y] = self._kriging.simulation(
                    coordinate,
                    sampled,
                    neighbor=neighbor,
                )

                # If any simulated value over the limit than discard that realization
                if self._grid[x][y] >= self._max_value or self._grid[x][y] <= self._min_value:
                    drop = True
                    iteration_failed += 1
                    if iteration_failed >= self.iteration_limit:
                        raise IterationError()
                    break

                # The four index value in sampled are [x, y, value, distance]
                sampled = (
                    np.array([(x, y, self._grid[x][y], 0)])
                    if len(sampled) == 0
                    else np.vstack((sampled, (x, y, self._grid[x][y], 0)))
                )

                if neighbor < self._max_neighbor:
                    neighbor += 1

            # IF drop is False then proceed to next realization generation
            if drop is False:
                self.random_fields[counts, :] = self._grid
                counts = counts + 1
                iteration_failed = 0
                # Set cov_cache flag to False to ensure cov_cache only compute once
                self.kriging._cov_cache_flag = False
            print('Progress = %.2f' % (counts / self.realization_number * 100) + '%', end='\r')

        print('Progress = %.2f' % 100 + '%\n', end='\r')
        end_time = time.time()
        print('Time = %f' % (end_time - start_time), 's\n')

        return self.random_fields

    def _read_shared_lib(self) -> CDLL:
        if sys.platform.startswith('linux'):
            lib = CDLL(str(BASE_DIR) + r'/c_core/uc_sgsim.so')
        elif sys.platform.startswith('win32'):
            lib = CDLL(str(BASE_DIR) + r'/c_core/uc_sgsim.dll', winmode=0)
        return lib

    def _run_c(self, n_process: int, randomseed: int) -> np.array:
        """
        Generate random field realizations with DLL support.

        Args:
            n_process (int): Number of parallel processes to use.
            randomseed (int): Seed for random number generation.

        Returns:
            np.array: Array containing generated random field realizations.
        """
        # Create a pool of processes and prepare the necessary arguments.
        # Then, distribute realizations and arguments to each process.
        pool = Pool(processes=n_process)
        self.n_process = n_process
        if n_process > 1:
            self.realization_number = self.realization_number * n_process
        self.random_fields = np.empty([self.realization_number, self.x_size])
        rand_list = [randomseed + i for i in range(n_process)]

        # Run parallel computing with dynamic link lib
        _simulation = pool.starmap(self._simulation_c, zip(rand_list))
        pool.close()
        # Use pool.join() to measure the coverage of sub process
        pool.join()

        # Collect results into a Python-List (random_field)
        self._reshape_simulaiton(_simulation, n_process)

        return self.random_fields

    def _simulation_c(self, randomseed: int) -> np.array:
        """
        Perform simulation using the shared library.

        Args:
            randomseed (int): Seed for random number generation.

        Returns:
            np.array: Array containing generated random field realizations.
        """
        # Read dynamic link lib and prepare arguments
        lib = self._read_shared_lib()
        mlen = int(self.x_size)
        realization_number = int(self.realization_number // self.n_process)
        random_field = np.empty([realization_number, self.x_size])
        kriging = 1 if self.kriging == 'OrdinaryKriging' else 0

        # Create sgsim and cov structure for dynamic link lib input
        sgsim_s = SgsimStructure(
            x_len=mlen,
            realization_numbers=realization_number,
            randomseed=randomseed,
            kirging_method=kriging,
            if_alloc_memory=0,
            max_iteration=self.iteration_limit,
            array=(c_double * (mlen * realization_number))(),
            z_min=self._min_value,
            z_max=self._max_value,
        )
        cov_s = CovModelStructure(
            bw_l=self.model.bandwidth_len,
            bw_s=self.model.bandwidth_step,
            bw=self.model.bandwidth_len // self.model.bandwidth_step,
            max_neighbor=self.max_neighbor,
            use_cov_cache=0 if self.constant_path is False else 1,
            range=self.model.k_range,
            sill=self.model.sill,
            nugget=self.model.nugget,
        )

        # Run simulation with dynamic link lib
        sgsim = lib.sgsim_run
        sgsim.argtypes = (POINTER(SgsimStructure), POINTER(CovModelStructure), c_int)
        sgsim(sgsim_s, cov_s, 0)

        # Collect results into a Python-List (random_field)
        for i in range(realization_number):
            random_field[i, :] = sgsim_s.array[i * mlen : (i + 1) * mlen]
        return random_field

    def _get_variogram_c(self, n_process: int = 1) -> np.array:
        """
        Compute the variogram of the generated random field with DLL support.

        Args:
            n_process (int, optional): Number of parallel processes to use (default is 1).
        """
        # Create a pool of processes and prepare the necessary arguments.
        # Then, distribute realizations and arguments to each process.
        pool = Pool(processes=n_process)
        self.n_process = n_process
        self.variogram = np.empty([self.realization_number, len(self.bandwidth)])
        cpu_number = [i for i in range(self.n_process)]

        # Run parallel computing with dynamic link lib
        z = pool.starmap(self._compute_variogram_c, zip(cpu_number))
        pool.close()
        # Use pool.join() to measure the coverage of sub process
        pool.join()

        # Collect results into a Python-List (random_field)
        for i in range(n_process):
            for j in range(int(self.realization_number / n_process)):
                self.variogram[(j + int(i * self.realization_number / n_process)), :] = z[i][j, :]
        return self.variogram

    def _compute_variogram_c(self, n_process: int) -> np.array:
        # Read dynamic link lib and prepare arguments
        lib = self._read_shared_lib()
        mlen = int(self.x_size)
        realization_number = int(self.realization_number // self.n_process)
        vario_size = len(self.bandwidth)
        vario_array = (c_double * (vario_size))()
        random_field_array = (c_double * (mlen))()
        variogram = np.empty([realization_number, vario_size])

        # Set function arguments and return type for ctypes
        vario = lib.variogram
        vario.argtypes = (
            POINTER(c_double),
            POINTER(c_double),
            c_int,
            c_int,
            c_int,
        )
        vario.restype = None

        # Run variogram computing with dynamic link lib and save results as a Python-List.
        for i in range(realization_number):
            random_field_array[:] = self.random_fields[i + n_process * realization_number, :]
            vario(random_field_array, vario_array, mlen, vario_size, 1)
            variogram[i, :] = list(vario_array)

        return variogram

    def _reshape_simulaiton(self, process_results: list[np.ndarray], n_process: int):
        # Collect results into a Python-List (random_field)
        for i in range(n_process):
            start = int(i * self.realization_number / n_process)
            end = int((i + 1) * self.realization_number / n_process)
            self.random_fields[start:end] = process_results[i][: end - start, :]
