/**
 * @file sgsim.c
 * @brief Reentrant implementation of the native 1D SGSIM engine.
 *
 * Copyright (c) 2022 Zncl2222
 * License: MIT
 */

# include <float.h>
# include <limits.h>
# include <math.h>
# include <stdint.h>
# include <stdio.h>
# include <stdlib.h>

# include "../include/cov_model.h"
# include "../include/kriging.h"
# include "../include/native_array.h"
# include "../include/random_tools.h"
# include "../include/sgsim.h"

typedef struct {
    sgsim_int_array_t x_grid;
    sgsim_double_array_t simulation;
    sgsim_double_array_t solution_cache;
    sampling_state sampling;
    kriging_workspace_t kriging;
} sgsim_workspace_t;

static void sgsim_workspace_free(sgsim_workspace_t* workspace) {
    if (workspace == NULL) {
        return;
    }
    sgsim_int_array_free(&workspace->x_grid);
    sgsim_double_array_free(&workspace->simulation);
    sgsim_double_array_free(&workspace->solution_cache);
    sampling_state_free(&workspace->sampling);
    kriging_workspace_free(&workspace->kriging);
    *workspace = (sgsim_workspace_t){0};
}

static int sgsim_workspace_init(
    sgsim_workspace_t* workspace,
    int x_len,
    const cov_model_t* cov_model) {
    if (workspace == NULL || cov_model == NULL || x_len <= 0) {
        return 0;
    }
    *workspace = (sgsim_workspace_t){0};
    size_t cache_stride = (size_t)cov_model->max_neighbor + 1;
    if (cov_model->use_cov_cache
        && (size_t)x_len > SIZE_MAX / cache_stride) {
        return 0;
    }
    size_t cache_size = cov_model->use_cov_cache
        ? (size_t)x_len * cache_stride : 1;

    if (sgsim_int_array_init(
            &workspace->x_grid,
            (size_t)x_len) != SGSIM_NUMERIC_OK
        || sgsim_double_array_init(
            &workspace->simulation,
            (size_t)x_len) != SGSIM_NUMERIC_OK
        || sgsim_double_array_init(
            &workspace->solution_cache,
            cache_size) != SGSIM_NUMERIC_OK
        || sgsim_int_array_iota(&workspace->x_grid) != SGSIM_NUMERIC_OK
        || !sampling_state_init(&workspace->sampling, x_len)
        || !kriging_workspace_init(&workspace->kriging, x_len, cov_model)) {
        sgsim_workspace_free(workspace);
        return 0;
    }
    return 1;
}

static int covariance_model_is_valid(int kind) {
    return kind == COV_MODEL_GAUSSIAN
        || kind == COV_MODEL_EXPONENTIAL
        || kind == COV_MODEL_SPHERICAL;
}

static sgsim_status_t validate_arguments(
    const sgsim_t* sgsim,
    const cov_model_t* cov_model,
    int vario_flag) {
    if (sgsim == NULL || cov_model == NULL) {
        return SGSIM_STATUS_INVALID_ARGUMENT;
    }
    if (vario_flag != 0) {
        return SGSIM_STATUS_UNSUPPORTED;
    }
    if (sgsim->x_len <= 0 || sgsim->x_len > INT_MAX - 2
        || sgsim->randomseed < 0
        || sgsim->realization_numbers <= 0) {
        return SGSIM_STATUS_INVALID_ARGUMENT;
    }
    if (sgsim->kriging_method == SGSIM_KRIGING_ORDINARY) {
        return SGSIM_STATUS_UNSUPPORTED;
    }
    if (sgsim->kriging_method != SGSIM_KRIGING_SIMPLE) {
        return SGSIM_STATUS_INVALID_ARGUMENT;
    }
    if (sgsim->if_alloc_memory != 0 && sgsim->if_alloc_memory != 1) {
        return SGSIM_STATUS_INVALID_ARGUMENT;
    }
    if ((sgsim->if_alloc_memory == 1 && sgsim->array != NULL)
        || (sgsim->if_alloc_memory == 0 && sgsim->array == NULL)) {
        return SGSIM_STATUS_INVALID_ARGUMENT;
    }
    if ((sgsim->constant_path != 0 && sgsim->constant_path != 1)
        || (cov_model->use_cov_cache != 0 && cov_model->use_cov_cache != 1)) {
        return SGSIM_STATUS_INVALID_ARGUMENT;
    }
    if (sgsim->iteration_limit < 0 || cov_model->max_neighbor < 0) {
        return SGSIM_STATUS_INVALID_ARGUMENT;
    }
    if (cov_model->bw_l <= 0 || cov_model->bw_s <= 0 || cov_model->k_range <= 0.0) {
        return SGSIM_STATUS_INVALID_ARGUMENT;
    }
    if (!isfinite(cov_model->k_range)
        || !isfinite(cov_model->sill)
        || !isfinite(cov_model->nugget)
        || cov_model->sill <= 0.0
        || cov_model->nugget < 0.0
        || cov_model->nugget > cov_model->sill) {
        return SGSIM_STATUS_INVALID_ARGUMENT;
    }
    if (!covariance_model_is_valid(cov_model->kind)) {
        return SGSIM_STATUS_INVALID_ARGUMENT;
    }
    if (cov_model->use_cov_cache != 0 && !sgsim->constant_path) {
        return SGSIM_STATUS_INVALID_ARGUMENT;
    }
    if (isnan(sgsim->z_min) || isnan(sgsim->z_max)) {
        return SGSIM_STATUS_INVALID_ARGUMENT;
    }
    return SGSIM_STATUS_OK;
}

void sgsim_init_defaults(sgsim_t* sgsim) {
    if (sgsim == NULL) {
        return;
    }
    *sgsim = (sgsim_t){
        .kriging_method = SGSIM_KRIGING_SIMPLE,
        .iteration_limit = 10,
        .z_min = -INFINITY,
        .z_max = INFINITY,
    };
}

void set_sgsim_defaults(sgsim_t* sgsim, cov_model_t* cov_model) {
    set_cov_model_default(cov_model);
    sgsim->iteration_limit = sgsim->iteration_limit == 0 ? 10 : sgsim->iteration_limit;
}

static void load_cached_solution(
    sgsim_workspace_t* workspace,
    int path_index,
    int neighbor_count,
    size_t cache_stride) {
    size_t cache_offset = (size_t)path_index * cache_stride;
    for (int j = 0; j < neighbor_count; j++) {
        workspace->kriging.weights.data[j] =
            workspace->solution_cache.data[cache_offset + (size_t)j];
    }
    workspace->kriging.kriging_std =
        workspace->solution_cache.data[cache_offset + (size_t)neighbor_count];
}

static void store_cached_solution(
    sgsim_workspace_t* workspace,
    int path_index,
    int neighbor_count,
    size_t cache_stride) {
    size_t cache_offset = (size_t)path_index * cache_stride;
    for (int j = 0; j < neighbor_count; j++) {
        workspace->solution_cache.data[cache_offset + (size_t)j] =
            workspace->kriging.weights.data[j];
    }
    workspace->solution_cache.data[cache_offset + (size_t)neighbor_count] =
        workspace->kriging.kriging_std;
}

sgsim_status_t sgsim_run_checked(
    sgsim_t* sgsim,
    const cov_model_t* cov_model,
    int vario_flag) {
    sgsim_status_t status = validate_arguments(sgsim, cov_model, vario_flag);
    if (status != SGSIM_STATUS_OK) {
        return status;
    }

    cov_model_t resolved_model = *cov_model;
    set_sgsim_defaults(sgsim, &resolved_model);
    resolved_model.max_neighbor = resolved_model.max_neighbor > sgsim->x_len
        ? sgsim->x_len : resolved_model.max_neighbor;
    if (sgsim->z_min >= sgsim->z_max) {
        return SGSIM_STATUS_INVALID_ARGUMENT;
    }

    size_t x_len = (size_t)sgsim->x_len;
    size_t realization_count = (size_t)sgsim->realization_numbers;
    if (x_len > SIZE_MAX / realization_count
        || x_len * realization_count > SIZE_MAX / sizeof(double)) {
        return SGSIM_STATUS_INVALID_ARGUMENT;
    }

    if (sgsim->if_alloc_memory == 1) {
        sgsim->array = calloc(x_len * realization_count, sizeof(double));
        if (sgsim->array == NULL) {
            return SGSIM_STATUS_ALLOCATION_FAILED;
        }
    }

    sgsim_workspace_t workspace = {0};
    if (!sgsim_workspace_init(&workspace, sgsim->x_len, &resolved_model)) {
        if (sgsim->if_alloc_memory == 1) {
            free(sgsim->array);
            sgsim->array = NULL;
        }
        return SGSIM_STATUS_ALLOCATION_FAILED;
    }

    sgsim_rng_t rng_state;
    sgsim_rng_init(&rng_state, (unsigned int)sgsim->randomseed);
    int count = 0;
    int error_times = 0;
    size_t cache_stride = (size_t)resolved_model.max_neighbor + 1U;

    while (count < sgsim->realization_numbers) {
        workspace.sampling.currlen = 0;
        workspace.sampling.neighbor = 0;
        int rejected = 0;

        if (!sgsim->constant_path || count == 0) {
            randompath(workspace.x_grid.data, sgsim->x_len, &rng_state);
        }

        for (int i = 0; i < sgsim->x_len; i++) {
            int use_solution_cache = resolved_model.use_cov_cache && count > 0;
            if (use_solution_cache) {
                load_cached_solution(
                    &workspace,
                    i,
                    workspace.sampling.neighbor,
                    cache_stride);
            }

            sampling_state_update(&workspace.sampling, workspace.x_grid.data[i]);
            if (simple_kriging(
                    workspace.simulation.data,
                    &workspace.sampling,
                    &workspace.kriging,
                    &rng_state,
                    use_solution_cache) != 0) {
                status = SGSIM_STATUS_NUMERICAL_ERROR;
                goto cleanup;
            }

            if (resolved_model.use_cov_cache && count == 0) {
                store_cached_solution(
                    &workspace,
                    i,
                    workspace.sampling.neighbor,
                    cache_stride);
            }

            int grid_index = workspace.x_grid.data[i];
            double value = workspace.simulation.data[grid_index];
            if (!isfinite(value)) {
                status = SGSIM_STATUS_NUMERICAL_ERROR;
                goto cleanup;
            }
            if (value >= sgsim->z_max || value <= sgsim->z_min) {
                rejected = 1;
                break;
            }

            sgsim->array[grid_index + sgsim->x_len * count] = value;
            if (workspace.sampling.neighbor < resolved_model.max_neighbor) {
                workspace.sampling.neighbor++;
            }
            workspace.sampling.sampled.data[i] = grid_index;
            workspace.sampling.currlen++;
        }

        if (!rejected) {
            count++;
            error_times = 0;
        } else {
            error_times++;
            if (error_times >= sgsim->iteration_limit) {
                status = SGSIM_STATUS_ITERATION_LIMIT;
                goto cleanup;
            }
        }
    }

cleanup:
    sgsim_workspace_free(&workspace);
    return status;
}

void sgsim_run(sgsim_t* sgsim, cov_model_t* cov_model, int vario_flag) {
    (void)sgsim_run_checked(sgsim, cov_model, vario_flag);
}

const char* sgsim_status_message(sgsim_status_t status) {
    switch (status) {
        case SGSIM_STATUS_OK:
            return "success";
        case SGSIM_STATUS_INVALID_ARGUMENT:
            return "invalid native simulation argument";
        case SGSIM_STATUS_ALLOCATION_FAILED:
            return "native simulation memory allocation failed";
        case SGSIM_STATUS_ITERATION_LIMIT:
            return "native simulation reached the realization rejection limit";
        case SGSIM_STATUS_NUMERICAL_ERROR:
            return "native kriging produced an invalid numerical result";
        case SGSIM_STATUS_UNSUPPORTED:
            return "requested native operation is not supported";
        default:
            return "unknown native simulation error";
    }
}

void sgsim_t_free(sgsim_t* sgsim) {
    if (sgsim == NULL) {
        return;
    }
    if (sgsim->if_alloc_memory == 1) {
        free(sgsim->array);
    }
    sgsim->array = NULL;
}
