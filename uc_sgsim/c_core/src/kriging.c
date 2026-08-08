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
    *sampling = (sampling_state){0};
    if (x_grid_len <= 0 || (size_t)x_grid_len > SIZE_MAX / sizeof(*sampling->sampled)) {
        return 0;
    }
    sampling->sampled = calloc((size_t)x_grid_len, sizeof(*sampling->sampled));
    return sampling->sampled != NULL;
}

void sampling_state_free(sampling_state* sampling) {
    free(sampling->sampled);
    *sampling = (sampling_state){0};
}

void sampling_state_update(sampling_state* sampling, int unsampled_point) {
    sampling->unsampled_point = unsampled_point;
}

int kriging_workspace_init(
    kriging_workspace_t* workspace,
    int x_len,
    const cov_model_t* cov_model) {
    *workspace = (kriging_workspace_t){0};
    if (x_len <= 0 || cov_model->max_neighbor <= 0) {
        return 0;
    }

    size_t neighbor_count = (size_t)cov_model->max_neighbor;
    if (neighbor_count > SIZE_MAX / neighbor_count
        || neighbor_count * neighbor_count > SIZE_MAX / sizeof(double)) {
        return 0;
    }
    size_t matrix_size = neighbor_count * neighbor_count;
    if ((size_t)x_len > SIZE_MAX / sizeof(*workspace->candidates)) {
        return 0;
    }

    workspace->model = cov_model;
    workspace->matrix_stride = cov_model->max_neighbor;
    workspace->candidates = calloc((size_t)x_len, sizeof(*workspace->candidates));
    workspace->covariance_vector = calloc(neighbor_count, sizeof(double));
    workspace->covariance_matrix = calloc(matrix_size, sizeof(double));
    workspace->factor_matrix = calloc(matrix_size, sizeof(double));
    workspace->weights = calloc(neighbor_count, sizeof(double));
    workspace->solve_temp = calloc(neighbor_count, sizeof(double));
    workspace->unit_solution = calloc(neighbor_count, sizeof(double));

    if (workspace->candidates == NULL
        || workspace->covariance_vector == NULL
        || workspace->covariance_matrix == NULL
        || workspace->factor_matrix == NULL
        || workspace->weights == NULL
        || workspace->solve_temp == NULL
        || workspace->unit_solution == NULL) {
        kriging_workspace_free(workspace);
        return 0;
    }
    return 1;
}

static int cholesky_attempt(
    kriging_workspace_t* workspace,
    int size,
    double jitter,
    double minimum_pivot) {
    int stride = workspace->matrix_stride;
    for (int row = 0; row < size; row++) {
        for (int column = 0; column < size; column++) {
            double value = workspace->covariance_matrix[row * stride + column];
            workspace->factor_matrix[row * stride + column] =
                row == column ? value + jitter : value;
        }
    }

    for (int row = 0; row < size; row++) {
        for (int column = 0; column <= row; column++) {
            double value = workspace->factor_matrix[row * stride + column];
            for (int index = 0; index < column; index++) {
                value -= workspace->factor_matrix[row * stride + index]
                    * workspace->factor_matrix[column * stride + index];
            }

            if (row == column) {
                if (!isfinite(value) || value <= minimum_pivot) {
                    return 0;
                }
                workspace->factor_matrix[row * stride + column] = sqrt(value);
            } else {
                double diagonal = workspace->factor_matrix[column * stride + column];
                workspace->factor_matrix[row * stride + column] = value / diagonal;
            }
        }
    }
    return 1;
}

static int factor_covariance(kriging_workspace_t* workspace, int size) {
    int stride = workspace->matrix_stride;
    double scale = DBL_MIN;
    for (int index = 0; index < size; index++) {
        scale = fmax(scale, fabs(workspace->covariance_matrix[index * stride + index]));
    }
    double minimum_pivot = scale * DBL_EPSILON * fmax(size, 1) * 16.0;

    workspace->diagonal_jitter = 0.0;
    if (cholesky_attempt(workspace, size, 0.0, minimum_pivot)) {
        return 1;
    }

    double jitter = minimum_pivot;
    for (int attempt = 0; attempt < 7; attempt++) {
        if (cholesky_attempt(workspace, size, jitter, minimum_pivot)) {
            workspace->diagonal_jitter = jitter;
            return 1;
        }
        jitter *= 10.0;
    }
    return 0;
}

static int solve_factor(
    kriging_workspace_t* workspace,
    const double* right_hand_side,
    int unit_right_hand_side,
    double* solution,
    int size) {
    int stride = workspace->matrix_stride;

    for (int row = 0; row < size; row++) {
        double value = unit_right_hand_side ? 1.0 : right_hand_side[row];
        for (int column = 0; column < row; column++) {
            value -= workspace->factor_matrix[row * stride + column]
                * workspace->solve_temp[column];
        }
        value /= workspace->factor_matrix[row * stride + row];
        if (!isfinite(value)) {
            return 0;
        }
        workspace->solve_temp[row] = value;
    }

    for (int row = size - 1; row >= 0; row--) {
        double value = workspace->solve_temp[row];
        for (int column = row + 1; column < size; column++) {
            value -= workspace->factor_matrix[column * stride + row] * solution[column];
        }
        value /= workspace->factor_matrix[row * stride + row];
        if (!isfinite(value)) {
            return 0;
        }
        solution[row] = value;
    }
    return 1;
}

static void prepare_neighbors(
    const double* array,
    const sampling_state* sampling,
    kriging_workspace_t* workspace) {
    for (int index = 0; index < sampling->currlen; index++) {
        int grid_index = sampling->sampled[index];
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
    int stride = workspace->matrix_stride;
    for (int row = 0; row < neighbor_count; row++) {
        workspace->covariance_vector[row] = cov_model_at_lag(
            workspace->candidates[row].distance,
            workspace->model);
        workspace->covariance_matrix[row * stride + row] = workspace->model->sill;

        for (int column = 0; column < row; column++) {
            double distance = fabs(
                (double)workspace->candidates[row].grid_index
                - workspace->candidates[column].grid_index);
            double covariance = cov_model_at_lag(distance, workspace->model);
            workspace->covariance_matrix[row * stride + column] = covariance;
            workspace->covariance_matrix[column * stride + row] = covariance;
        }
    }
}

static int solve_kriging_system(
    kriging_workspace_t* workspace,
    int kriging_method,
    int neighbor_count) {
    build_covariance_system(workspace, neighbor_count);
    if (!factor_covariance(workspace, neighbor_count)) {
        return 0;
    }
    if (!solve_factor(
            workspace,
            workspace->covariance_vector,
            0,
            workspace->weights,
            neighbor_count)) {
        return 0;
    }

    double lagrange_multiplier = 0.0;
    if (kriging_method == 1) {
        if (!solve_factor(
                workspace,
                NULL,
                1,
                workspace->unit_solution,
                neighbor_count)) {
            return 0;
        }

        double weight_sum = 0.0;
        double unit_sum = 0.0;
        for (int index = 0; index < neighbor_count; index++) {
            weight_sum += workspace->weights[index];
            unit_sum += workspace->unit_solution[index];
        }
        if (!isfinite(unit_sum) || fabs(unit_sum) <= DBL_MIN) {
            return 0;
        }
        lagrange_multiplier = (weight_sum - 1.0) / unit_sum;
        for (int index = 0; index < neighbor_count; index++) {
            workspace->weights[index] -=
                lagrange_multiplier * workspace->unit_solution[index];
        }
    }

    double explained_variance = 0.0;
    for (int index = 0; index < neighbor_count; index++) {
        explained_variance +=
            workspace->weights[index] * workspace->covariance_vector[index];
    }
    double kriging_variance =
        workspace->model->sill - explained_variance - lagrange_multiplier;
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
    int kriging_method,
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
        && !solve_kriging_system(workspace, kriging_method, sampling->neighbor)) {
        return 1;
    }

    double estimation = 0.0;
    for (int index = 0; index < sampling->neighbor; index++) {
        if (!isfinite(workspace->weights[index])) {
            return 1;
        }
        estimation += workspace->candidates[index].value * workspace->weights[index];
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
    free(workspace->candidates);
    free(workspace->covariance_vector);
    free(workspace->covariance_matrix);
    free(workspace->factor_matrix);
    free(workspace->weights);
    free(workspace->solve_temp);
    free(workspace->unit_solution);
    *workspace = (kriging_workspace_t){0};
}
