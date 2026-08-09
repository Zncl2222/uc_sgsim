/**
 * @file kriging.c
 * @brief Allocation-free hot path for native 1D Kriging.
 *
 * Copyright (c) 2022-2023 Zncl2222
 * License: MIT
 */

# include <float.h>
# include <math.h>
# include <stdint.h>
# include <stdlib.h>

# include "../include/cov_model.h"
# include "../include/kriging.h"
# include "../include/native_array.h"
# include "../include/native_matrix.h"
# include "../include/random_tools.h"

static void swap_neighbor(kriging_neighbor_t* left, kriging_neighbor_t* right) {
    kriging_neighbor_t temporary = *left;
    *left = *right;
    *right = temporary;
}

static int neighbor_precedes(
    const kriging_neighbor_t* left,
    const kriging_neighbor_t* right) {
    if (left->distance < right->distance) {
        return 1;
    }
    if (left->distance > right->distance) {
        return 0;
    }
    return left->grid_index < right->grid_index;
}

static int partition_neighbors(kriging_neighbor_t* candidates, int left, int right) {
    int pivot_index = left + (right - left) / 2;
    swap_neighbor(&candidates[pivot_index], &candidates[right]);
    kriging_neighbor_t pivot = candidates[right];
    int store_index = left;

    for (int index = left; index < right; index++) {
        if (neighbor_precedes(&candidates[index], &pivot)) {
            swap_neighbor(&candidates[index], &candidates[store_index]);
            store_index++;
        }
    }
    swap_neighbor(&candidates[store_index], &candidates[right]);
    return store_index;
}

static void select_nearest_neighbors(
    kriging_neighbor_t* candidates,
    int candidate_count,
    int neighbor_count) {
    if (neighbor_count >= candidate_count) {
        return;
    }

    int left = 0;
    int right = candidate_count - 1;
    int target = neighbor_count - 1;
    while (left < right) {
        int pivot = partition_neighbors(candidates, left, right);
        if (pivot == target) {
            return;
        }
        if (pivot > target) {
            right = pivot - 1;
        } else {
            left = pivot + 1;
        }
    }
}

int sampling_state_init(sampling_state* sampling, int x_grid_len) {
    if (sampling == NULL) {
        return 0;
    }
    *sampling = (sampling_state){0};
    if (x_grid_len <= 0) {
        return 0;
    }
    return sgsim_int_array_init(&sampling->sampled, (size_t)x_grid_len)
        == SGSIM_NUMERIC_OK;
}

void sampling_state_free(sampling_state* sampling) {
    if (sampling == NULL) {
        return;
    }
    sgsim_int_array_free(&sampling->sampled);
    *sampling = (sampling_state){0};
}

void sampling_state_update(sampling_state* sampling, int unsampled_point) {
    sampling->unsampled_point = unsampled_point;
}

int kriging_workspace_init(
    kriging_workspace_t* workspace,
    int x_len,
    const cov_model_t* cov_model) {
    if (workspace == NULL || cov_model == NULL) {
        return 0;
    }
    *workspace = (kriging_workspace_t){0};
    if (x_len <= 0 || cov_model->max_neighbor < 0) {
        return 0;
    }

    workspace->model = cov_model;
    if (cov_model->max_neighbor == 0) {
        return 1;
    }

    size_t neighbor_count = (size_t)cov_model->max_neighbor;
    if (neighbor_count > SIZE_MAX / neighbor_count
        || neighbor_count * neighbor_count > SIZE_MAX / sizeof(double)) {
        return 0;
    }
    if ((size_t)x_len > SIZE_MAX / sizeof(*workspace->candidates)) {
        return 0;
    }

    workspace->candidates = calloc((size_t)x_len, sizeof(*workspace->candidates));
    if (workspace->candidates == NULL
        || sgsim_double_array_init(
            &workspace->covariance_vector,
            neighbor_count) != SGSIM_NUMERIC_OK
        || sgsim_double_matrix_init(
            &workspace->covariance_matrix,
            neighbor_count,
            neighbor_count) != SGSIM_NUMERIC_OK
        || sgsim_double_matrix_init(
            &workspace->factor_matrix,
            neighbor_count,
            neighbor_count) != SGSIM_NUMERIC_OK
        || sgsim_double_array_init(
            &workspace->weights,
            neighbor_count) != SGSIM_NUMERIC_OK
        || sgsim_double_array_init(
            &workspace->solve_temp,
            neighbor_count) != SGSIM_NUMERIC_OK) {
        kriging_workspace_free(workspace);
        return 0;
    }
    return 1;
}

static void prepare_neighbors(
    const double* array,
    const sampling_state* sampling,
    kriging_workspace_t* workspace) {
    for (int index = 0; index < sampling->currlen; index++) {
        int grid_index = sampling->sampled.data[index];
        workspace->candidates[index] = (kriging_neighbor_t){
            .grid_index = grid_index,
            .value = array[grid_index],
            .distance = fabs((double)grid_index - sampling->unsampled_point),
        };
    }
    select_nearest_neighbors(
        workspace->candidates,
        sampling->currlen,
        sampling->neighbor);
}

static void build_covariance_system(
    kriging_workspace_t* workspace,
    int neighbor_count) {
    size_t stride = workspace->covariance_matrix.stride;
    for (int row = 0; row < neighbor_count; row++) {
        workspace->covariance_vector.data[row] = cov_model_at_lag(
            workspace->candidates[row].distance,
            workspace->model);
        workspace->covariance_matrix.data[(size_t)row * stride + (size_t)row] =
            workspace->model->sill;

        for (int column = 0; column < row; column++) {
            double distance = fabs(
                (double)workspace->candidates[row].grid_index
                - workspace->candidates[column].grid_index);
            double covariance = cov_model_at_lag(distance, workspace->model);
            workspace->covariance_matrix.data[
                (size_t)row * stride + (size_t)column] = covariance;
            workspace->covariance_matrix.data[
                (size_t)column * stride + (size_t)row] = covariance;
        }
    }
}

static int solve_kriging_system(
    kriging_workspace_t* workspace,
    int neighbor_count) {
    build_covariance_system(workspace, neighbor_count);
    if (sgsim_cholesky_factorize_regularized(
            &workspace->covariance_matrix,
            &workspace->factor_matrix,
            (size_t)neighbor_count,
            &workspace->diagonal_jitter) != SGSIM_NUMERIC_OK) {
        return 0;
    }
    if (sgsim_cholesky_solve(
            &workspace->factor_matrix,
            workspace->covariance_vector.data,
            workspace->solve_temp.data,
            workspace->weights.data,
            (size_t)neighbor_count) != SGSIM_NUMERIC_OK) {
        return 0;
    }

    double explained_variance = 0.0;
    if (sgsim_vector_dot(
            workspace->weights.data,
            workspace->covariance_vector.data,
            (size_t)neighbor_count,
            &explained_variance) != SGSIM_NUMERIC_OK) {
        return 0;
    }
    double kriging_variance = workspace->model->sill - explained_variance;
    if (workspace->diagonal_jitter > 0.0) {
        double weighted_covariance = 0.0;
        if (sgsim_symmetric_quadratic_form(
                &workspace->covariance_matrix,
                workspace->weights.data,
                (size_t)neighbor_count,
                &weighted_covariance) != SGSIM_NUMERIC_OK) {
            return 0;
        }
        kriging_variance = workspace->model->sill
            - 2.0 * explained_variance + weighted_covariance;
    }
    double tolerance = fmax(fabs(workspace->model->sill), DBL_MIN) * 1e-10;
    if (!isfinite(kriging_variance) || kriging_variance < -tolerance) {
        return 0;
    }
    workspace->kriging_std = sqrt(fmax(kriging_variance, 0.0));
    return 1;
}

int simple_kriging(
    double* array,
    sampling_state* sampling,
    kriging_workspace_t* workspace,
    sgsim_rng_t* rng_state,
    int use_solution_cache) {
    if (sampling->neighbor == 0) {
        workspace->kriging_std = sqrt(workspace->model->sill);
        double value = sgsim_random_normal(rng_state) * workspace->kriging_std;
        if (!isfinite(value)) {
            return 1;
        }
        array[sampling->unsampled_point] = value;
        return 0;
    }

    prepare_neighbors(array, sampling, workspace);
    if (!use_solution_cache
        && !solve_kriging_system(workspace, sampling->neighbor)) {
        return 1;
    }

    double estimation = 0.0;
    for (int index = 0; index < sampling->neighbor; index++) {
        if (!isfinite(workspace->weights.data[index])) {
            return 1;
        }
        estimation += workspace->candidates[index].value
            * workspace->weights.data[index];
    }
    if (!isfinite(estimation)
        || !isfinite(workspace->kriging_std)
        || workspace->kriging_std < 0.0) {
        return 1;
    }

    double value = estimation + sgsim_random_normal(rng_state) * workspace->kriging_std;
    if (!isfinite(value)) {
        return 1;
    }
    array[sampling->unsampled_point] = value;
    return 0;
}

void kriging_workspace_free(kriging_workspace_t* workspace) {
    if (workspace == NULL) {
        return;
    }
    free(workspace->candidates);
    sgsim_double_array_free(&workspace->covariance_vector);
    sgsim_double_matrix_free(&workspace->covariance_matrix);
    sgsim_double_matrix_free(&workspace->factor_matrix);
    sgsim_double_array_free(&workspace->weights);
    sgsim_double_array_free(&workspace->solve_temp);
    *workspace = (kriging_workspace_t){0};
}
