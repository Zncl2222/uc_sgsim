/**
 * @file kriging.c
 * @brief Implementation of Kriging functions in the SGSIM library.
 *
 * This file contains the implementation of functions related to Kriging,
 * used in the Sequential Gaussian Simulation (SGSIM) library. Kriging is
 * employed for spatial interpolation and estimation of values.
 *
 * Copyright (c) 2022-2023 Zncl2222
 * License: MIT
 */

# include <stdio.h>
# include <stdlib.h>
# include <math.h>
# include <float.h>
# include "../include/kriging.h"
# include "../include/random_tools.h"
# include "../include/cov_model.h"
# include "../include/matrix_tools.h"
# include "../include/sort_tools.h"
# include "../c_array_tools/src/c_array.h"

int sampling_state_init(sampling_state* sampling, int x_grid_len) {
    sampling->neighbor = 0;
    sampling->currlen = 0;
    sampling->idx = 0;
    sampling->unsampled_point = 0;

    c_array_init(&sampling->sampled, x_grid_len);
    c_array_init(&sampling->u_array, x_grid_len);

    if (sampling->sampled.data == NULL || sampling->u_array.data == NULL) {
        sampling_state_free(sampling);
        return 0;
    }
    return 1;
}

void sampling_state_free(sampling_state* sampling) {
    free(sampling->sampled.data);
    free(sampling->u_array.data);
    sampling->sampled.data = NULL;
    sampling->u_array.data = NULL;
}

void sampling_state_update(sampling_state* sampling, double unsampled_point, int idx) {
    sampling->unsampled_point = unsampled_point;
    sampling->idx = idx;
}

int kriging_workspace_init(
    kriging_workspace_t* workspace,
    int x_len,
    const cov_model_t* cov_model) {
    *workspace = (kriging_workspace_t){0};
    workspace->model = cov_model;
    int buffer = cov_model->max_neighbor + 2;
    c_array_init(&workspace->location, buffer);
    c_array_init(&workspace->location_cov, buffer);
    c_array_init(&workspace->location_cov2d, buffer);
    c_array_init(&workspace->weights, buffer);
    c_array_init(&workspace->flatten_temp, buffer * buffer);

    if (workspace->location.data == NULL
        || workspace->location_cov.data == NULL
        || workspace->location_cov2d.data == NULL
        || workspace->weights.data == NULL
        || workspace->flatten_temp.data == NULL) {
        kriging_workspace_free(workspace);
        return 0;
    }

    c_matrix_init(&workspace->pdist_temp, buffer, buffer);
    c_matrix_init(&workspace->data_cov, buffer, buffer);
    c_matrix_init(&workspace->distance_mat, x_len, 3);
    return 1;
}

int simple_kriging(
    double* array,
    sampling_state* sampling,
    kriging_workspace_t* workspace,
    mt19937_state* rng_state,
    int kriging_method,
    int use_cov_cache) {
    int has_neighbor = find_neighbor(array, sampling, workspace, rng_state);

    if (has_neighbor == 0) {
        return 0;
    }

    for (int j = 0; j < sampling->currlen; j++) {
        workspace->distance_mat.data[j][0] = sampling->sampled.data[j];
        workspace->distance_mat.data[j][1] = array[(int)sampling->sampled.data[j]];
        workspace->distance_mat.data[j][2] = sampling->u_array.data[j];
    }

    if (sampling->neighbor >= 2) {
        quickselect2d(
            workspace->distance_mat.data,
            0,
            sampling->currlen - 1,
            sampling->neighbor);
    }

    for (int j = 0; j < sampling->neighbor; j++) {
        workspace->location.data[j] = workspace->distance_mat.data[j][0];
        if (use_cov_cache == 0) {
            workspace->location_cov2d.data[j] = workspace->distance_mat.data[j][2];
        }
    }
    if (use_cov_cache == 0) {
        cov_compute(
            workspace->location_cov2d.data,
            workspace->location_cov.data,
            sampling->neighbor,
            workspace->model);
    }
    pdist(workspace->location.data, workspace->pdist_temp.data, sampling->neighbor);
    cov_compute2d(
        workspace->pdist_temp.data,
        workspace->flatten_temp.data,
        sampling->neighbor,
        workspace->model);
    matrixform(
        workspace->flatten_temp.data,
        workspace->data_cov.data,
        sampling->neighbor);

    if (kriging_method == 1) {
        matrix_augmented(workspace->data_cov.data, sampling->neighbor);
        workspace->location_cov.data[sampling->neighbor] = 1.0;
    }

    int neighbor = kriging_method == 1 ? sampling->neighbor + 1 : sampling->neighbor;
    if (sampling->neighbor >= 1)
        lu_inverse_solver(
            workspace->data_cov.data,
            workspace->location_cov.data,
            workspace->weights.data,
            neighbor);

    double estimation = 0.0;
    double kriging_var = 0.0;

    for (int j = 0; j < sampling->neighbor; j++) {
        estimation += workspace->distance_mat.data[j][1] * workspace->weights.data[j];
        kriging_var += workspace->location_cov.data[j] * workspace->weights.data[j];
    }

    kriging_var = workspace->model->sill - kriging_var;
    if (kriging_method == 1) {
        kriging_var -= workspace->weights.data[sampling->neighbor];
    }
    double tolerance = fmax(fabs(workspace->model->sill), DBL_MIN) * 1e-10;
    if (!isfinite(estimation) || !isfinite(kriging_var) || kriging_var < -tolerance) {
        return 1;
    }
    kriging_var = fmax(kriging_var, 0.0);
    double fix = random_normal(rng_state) * sqrt(kriging_var);

    array[(int)sampling->unsampled_point] = estimation + fix;
    return isfinite(array[(int)sampling->unsampled_point]) ? 0 : 1;
}

int find_neighbor(
    double* array,
    sampling_state* sampling,
    const kriging_workspace_t* workspace,
    mt19937_state* rng_state) {
    if (sampling->neighbor == 0) {
        array[(int)sampling->unsampled_point] =
            random_normal(rng_state) * sqrt(workspace->model->sill);
        sampling->sampled.data[sampling->idx] = sampling->unsampled_point;
        return 0;
    }

    for (int j = 0; j < sampling->currlen; j++) {
        sampling->u_array.data[j] = fabs(sampling->sampled.data[j] - sampling->unsampled_point);
    }

    return 1;
}

void matrix_augmented(double** mat, int neighbor) {
    for (int i = 0; i < neighbor; i++) {
        mat[i][neighbor] = 1;
    }
    for (int i = 0; i <= neighbor; i++) {
        if (i == (neighbor)) {
            mat[neighbor][i] = 0;
        } else {
            mat[neighbor][i] = 1;
        }
    }
}

void kriging_workspace_free(kriging_workspace_t* workspace) {
    free(workspace->location.data);
    free(workspace->location_cov.data);
    free(workspace->location_cov2d.data);
    free(workspace->flatten_temp.data);
    free(workspace->weights.data);
    if (workspace->distance_mat.data != NULL) {
        c_matrix_free(&workspace->distance_mat);
    }
    if (workspace->pdist_temp.data != NULL) {
        c_matrix_free(&workspace->pdist_temp);
    }
    if (workspace->data_cov.data != NULL) {
        c_matrix_free(&workspace->data_cov);
    }
    *workspace = (kriging_workspace_t){0};
}
