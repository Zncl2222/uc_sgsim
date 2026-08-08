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

# include <stdio.h>
# include <math.h>
# include <stdlib.h>
# include <string.h>
# include <errno.h>
# include "../include/matrix_tools.h"
# include "../c_array_tools/src/c_array.h"
# ifdef __WIN32__
# include <direct.h>
# include <io.h>
# elif defined(__linux__) || defined(__unix__)
# include <fcntl.h>
# include <sys/io.h>
# include <sys/stat.h>
struct stat st = {0};
# endif


void lu_inverse_solver(double** mat, const double* array, double* result, int n) {
    c_matrix_double lower;
    c_matrix_double upper;
    c_array_double y;

    int buffer = n + 1;
    c_matrix_init(&lower, buffer, buffer);
    c_matrix_init(&upper, buffer, buffer);
    c_array_init(&y, buffer);

    lu_decomposition(mat, lower.data, upper.data, n);

    // Solve L(Ux)=b, assume Ux=y
    y.data[0] = array[0] / lower.data[0][0];  // NOSONAR

    for (int i = 1; i < n; i++) {
        y.data[i] = array[i];
        for (int j = 0; j < i; j++) {
            y.data[i] = y.data[i] - lower.data[i][j] * y.data[j];
        }
        y.data[i] = y.data[i] / lower.data[i][i];
    }

    result[n] = y.data[n];

    for (int i = n - 1; i >= 0; i--) {
        result[i] = y.data[i];
        for (int j = i + 1; j < n; j++) {
            result[i] -= upper.data[i][j] * result[j];
        }
    }

    c_matrix_free(&lower);
    c_matrix_free(&upper);
    c_array_free(&y);
}

void lu_decomposition(double** mat, double** l, double** u, int n) {
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
                u[i][j] = mat[i][j] / l[i][i];
                for (int k = 0; k < i; k++) {
                    u[i][j] = u[i][j] - ((l[i][k] * u[k][j]) / l[i][i]);
                }
            }
        }
    }
}

int* arange(int x) {
    int* res;
    res = (int*)malloc(x*sizeof(int));

    for (int i = 0; i < x; i++) {
        res[i] = i;
    }
    return res;
}

double* d_arange(int x) {
    double* space;
    space = (double*)malloc(x*sizeof(double));

    for (int i = 0; i < x; i++) {
        space[i] = i;
    }
    return space;
}

void pdist(const double* x, double** c, int n_dim) {
    for (int i = 0; i < n_dim; i++) {
        for (int j = 0; j < n_dim; j++) {
            c[i][j] = fabs(x[j] - x[i]);
        }
    }
}

void matrixform(const double* x, double** matrix, int n_dim) {
    int index = 0;
    for (int i = 0; i < n_dim; i++) {
        for (int j = 0; j < n_dim; j++) {
            matrix[i][j] = x[index];
            index++;
        }
    }
}

void save_1darray(const double* array, int array_size,
                char* fhead, char* path , int total_n, int curr_n) {
    FILE *output;
    int num_digits = (int)ceil(log10(total_n)) + 1;
    char format[200];

    snprintf(format, sizeof(format), "%s%s%%0%dd.txt", path, fhead, num_digits);

    #ifdef __linux__
    if (mkdir(path, 0770 | O_EXCL) == -1) {
        if (errno != EEXIST) {
            printf("Folder already exist!");
        }
    }
    #elif defined(__WIN32__)
    if (_mkdir(path) == -1) {
        if (errno != EEXIST) {
            printf("Folder already exist!");
        }
    }
    #endif

    char filename[200];
    snprintf(filename, sizeof(filename), format, curr_n);

    output = fopen(filename, "w");
    if (output == NULL) {
        perror("Failed to open the file");
        exit(1);
    }

    for (int i = 0; i < array_size; i++) {
        fprintf(output, "%d\t%.10f\n", i, array[i]);
    }

    fclose(output);
}
