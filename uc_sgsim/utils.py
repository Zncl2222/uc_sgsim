import os
from ctypes import Structure, POINTER, c_double, c_int

import numpy as np


def _scalar_value(value) -> float:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError('Expected a scalar or size-one array')
    return float(array.reshape(-1)[0])


def save_as_multiple_file(
    number: str,
    size: int,
    field: list,
    file_type: str,
    fieldtype: str,
) -> None:
    """
    Save realizations to multiple files.

    Args:
        number (str): A string representing the file number.
        size (int): The size of the data.
        field (list): The data to be saved.
        file_type (str): The file extension (e.g., 'txt', 'csv').
        fieldtype (str): The file and folder name.

    Returns:
        None
    """
    idx = number.strip('0')
    os.makedirs(f'./{fieldtype}', exist_ok=True)
    idx = 0 if not idx else int(idx)
    with open(f'./{fieldtype}/{fieldtype}{number}.{file_type}', 'w') as f:
        for j in range(0, size):
            print(
                '%.2d' % (j),
                '%10.6f' % (_scalar_value(field[idx, j])),
                file=f,
            )


def save_as_one_file(path: str, field: list) -> None:
    """
    Save realizations to a single file.

    Args:
        path (str): The path of the output file.
        field (list): The data to be saved.

    Returns:
        None
    """
    with open(f'{path}', 'w') as f:
        for i in range(len(field[:, 0])):
            for j in range(len(field[0, :])):
                end = '\n' if j == len(field[0, :]) - 1 else ', '
                print(
                    '%10.6f' % (_scalar_value(field[i, j])),
                    file=f,
                    end=end,
                )


class SgsimStructure(Structure):
    """
    Define a structure for geostatistical simulations using ctypes.

    This structure is used for interfacing with a C library.
    """

    _fields_ = [
        ('x_len', c_int),
        ('realization_numbers', c_int),
        ('randomseed', c_int),
        ('kriging_method', c_int),
        ('if_alloc_memory', c_int),
        ('max_iteration', c_int),
        ('array', POINTER(c_double)),
        ('z_min', c_double),
        ('z_max', c_double),
    ]


class CovModelStructure(Structure):
    """
    Define a structure for geostatistical simulations using ctypes.

    This structure is used for interfacing with a C library.
    """

    _fields_ = [
        ('bw_l', c_int),
        ('bw_s', c_int),
        ('bw', c_int),
        ('max_neighbor', c_int),
        ('use_cov_cache', c_int),
        ('range', c_double),
        ('sill', c_double),
        ('nugget', c_double),
    ]
