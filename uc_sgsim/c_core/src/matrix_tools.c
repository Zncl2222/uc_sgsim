/**
 * @file matrix_tools.c
 * @brief Implementation of Matrix and Array Manipulation Functions
 *
 * This source file contains the implementation of various matrix and array manipulation functions.
 * These functions provide utilities for tasks such as LU decomposition, array generation, and file
 * output of arrays.
 *
 * Copyright (c) 2022-2023 Zncl2222
 * License: MIT
 */

# include <errno.h>
# include <math.h>
# include <stdio.h>
# include <stdlib.h>
# include <string.h>

# include "../include/matrix_tools.h"
# include "../include/native_array.h"
# include "../include/native_matrix.h"

# ifdef _WIN32
# include <direct.h>
# else
# include <sys/stat.h>
# endif

static void mark_solver_failure(double* result, int n) {
    if (result == NULL || n <= 0) {
        return;
    }
    for (int index = 0; index < n; index++) {
        result[index] = NAN;
    }
}

void lu_inverse_solver(double** mat, const double* array, double* result, int n) {
    if (mat == NULL || array == NULL || result == NULL || n <= 0) {
        mark_solver_failure(result, n);
        return;
    }
    for (int row = 0; row < n; row++) {
        if (mat[row] == NULL) {
            mark_solver_failure(result, n);
            return;
        }
    }

    sgsim_double_matrix_t lower = {0};
    sgsim_double_matrix_t upper = {0};
    sgsim_double_array_t temporary = {0};
    size_t order = (size_t)n;
    if (sgsim_double_matrix_init(&lower, order, order) != SGSIM_NUMERIC_OK
        || sgsim_double_matrix_init(&upper, order, order) != SGSIM_NUMERIC_OK
        || sgsim_double_array_init(&temporary, order) != SGSIM_NUMERIC_OK) {
        mark_solver_failure(result, n);
        goto cleanup;
    }

    for (int pivot = 0; pivot < n; pivot++) {
        for (int row = pivot; row < n; row++) {
            double value = mat[row][pivot];
            for (int index = 0; index < pivot; index++) {
                value -= lower.data[(size_t)row * lower.stride + (size_t)index]
                    * upper.data[(size_t)index * upper.stride + (size_t)pivot];
            }
            lower.data[(size_t)row * lower.stride + (size_t)pivot] = value;
        }
        double diagonal = lower.data[(size_t)pivot * lower.stride + (size_t)pivot];
        if (!isfinite(diagonal) || fabs(diagonal) <= 1e-15) {
            mark_solver_failure(result, n);
            goto cleanup;
        }
        upper.data[(size_t)pivot * upper.stride + (size_t)pivot] = 1.0;
        for (int column = pivot + 1; column < n; column++) {
            double value = mat[pivot][column];
            for (int index = 0; index < pivot; index++) {
                value -= lower.data[(size_t)pivot * lower.stride + (size_t)index]
                    * upper.data[(size_t)index * upper.stride + (size_t)column];
            }
            upper.data[(size_t)pivot * upper.stride + (size_t)column] =
                value / diagonal;
        }
    }

    for (int row = 0; row < n; row++) {
        double value = array[row];
        for (int column = 0; column < row; column++) {
            value -= lower.data[(size_t)row * lower.stride + (size_t)column]
                * temporary.data[column];
        }
        value /= lower.data[(size_t)row * lower.stride + (size_t)row];
        if (!isfinite(value)) {
            mark_solver_failure(result, n);
            goto cleanup;
        }
        temporary.data[row] = value;
    }

    for (int row = n; row-- > 0;) {
        double value = temporary.data[row];
        for (int column = row + 1; column < n; column++) {
            value -= upper.data[(size_t)row * upper.stride + (size_t)column]
                * result[column];
        }
        result[row] = value;
    }

cleanup:
    sgsim_double_matrix_free(&lower);
    sgsim_double_matrix_free(&upper);
    sgsim_double_array_free(&temporary);
}

void lu_decomposition(double** mat, double** l, double** u, int n) {
    if (mat == NULL || l == NULL || u == NULL || n <= 0) {
        return;
    }
    for (int row = 0; row < n; row++) {
        if (mat[row] == NULL || l[row] == NULL || u[row] == NULL) {
            return;
        }
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (j < i) {
                l[j][i] = 0;
            } else {
                l[j][i] = mat[j][i];  // NOSONAR
                for (int k = 0; k < i; k++) {
                    l[j][i] = l[j][i] - l[j][k] * u[k][i];
                }
            }
        }
        for (int j = 0; j < n; j++) {
            if (j < i) {
                u[i][j] = 0;
            } else if (j == i) {
                u[i][j] = 1;
            } else {
                if (fabs(l[i][i]) <= 1e-15) {
                    u[i][j] = NAN;
                    continue;
                }
                u[i][j] = mat[i][j] / l[i][i];
                for (int k = 0; k < i; k++) {
                    u[i][j] = u[i][j] - ((l[i][k] * u[k][j]) / l[i][i]);
                }
            }
        }
    }
}

int* arange(int x) {
    if (x <= 0) {
        return NULL;
    }
    sgsim_int_array_t array = {0};
    if (sgsim_int_array_init(&array, (size_t)x) != SGSIM_NUMERIC_OK
        || sgsim_int_array_iota(&array) != SGSIM_NUMERIC_OK) {
        sgsim_int_array_free(&array);
        return NULL;
    }
    return array.data;
}

double* d_arange(int x) {
    if (x <= 0) {
        return NULL;
    }
    sgsim_double_array_t array = {0};
    if (sgsim_double_array_init(&array, (size_t)x) != SGSIM_NUMERIC_OK) {
        return NULL;
    }
    for (int i = 0; i < x; i++) {
        array.data[i] = (double)i;
    }
    return array.data;
}

void pdist(const double* x, double** c, int n_dim) {
    if (x == NULL || c == NULL || n_dim <= 0) {
        return;
    }
    for (int i = 0; i < n_dim; i++) {
        if (c[i] == NULL) {
            return;
        }
        c[i][i] = 0.0;
        for (int j = 0; j < i; j++) {
            double distance = fabs(x[j] - x[i]);
            c[i][j] = distance;
            c[j][i] = distance;
        }
    }
}

void matrixform(const double* x, double** matrix, int n_dim) {
    if (x == NULL || matrix == NULL || n_dim <= 0) {
        return;
    }
    size_t index = 0;
    for (int i = 0; i < n_dim; i++) {
        if (matrix[i] == NULL) {
            return;
        }
        for (int j = 0; j < n_dim; j++) {
            matrix[i][j] = x[index];
            index++;
        }
    }
}

void save_1darray(const double* array, int array_size,
                const char* fhead, const char* path, int total_n, int curr_n) {
    if (array == NULL || array_size < 0 || fhead == NULL || path == NULL) {
        return;
    }
    int num_digits = total_n > 1 ? (int)floor(log10((double)total_n)) + 1 : 1;
    #ifdef _WIN32
    if (_mkdir(path) == -1) {
        if (errno != EEXIST) {
            perror("Failed to create output directory");
            return;
        }
    }
    #else
    if (mkdir(path, 0770) == -1 && errno != EEXIST) {
        perror("Failed to create output directory");
        return;
    }
    #endif

    char filename[200];
    int filename_length = snprintf(
        filename,
        sizeof(filename),
        "%s%s%0*d.txt",
        path,
        fhead,
        num_digits,
        curr_n);
    if (filename_length < 0 || (size_t)filename_length >= sizeof(filename)) {
        return;
    }

    FILE* output = fopen(filename, "w");
    if (output == NULL) {
        perror("Failed to open the file");
        return;
    }

    for (int i = 0; i < array_size; i++) {
        fprintf(output, "%d\t%.10f\n", i, array[i]);
    }

    fclose(output);
}
