/**
 * @file sgsim.h
 * @brief Header file for Sequential Gaussian Simulation (SGSIM) library.
 *
 * This header defines the data structure sgsim_t and several functions for
 * conducting sequential Gaussian simulation.
 *
 * Copyright (c) 2022-2023 Zncl2222
 * License: MIT
 */

#ifndef UC_SGSIM_C_CORE_INCLUDE_SGSIM_H_
#define UC_SGSIM_C_CORE_INCLUDE_SGSIM_H_

# include "cov_model.h"
# include "../c_array_tools/src/c_array.h"

typedef enum {
    SGSIM_KRIGING_SIMPLE = 0,
    /** Reserved for ABI compatibility; unconditional OK is unsupported. */
    SGSIM_KRIGING_ORDINARY = 1
} sgsim_kriging_method_t;

typedef enum {
    SGSIM_STATUS_OK = 0,
    SGSIM_STATUS_INVALID_ARGUMENT = 1,
    SGSIM_STATUS_ALLOCATION_FAILED = 2,
    SGSIM_STATUS_ITERATION_LIMIT = 3,
    SGSIM_STATUS_NUMERICAL_ERROR = 4,
    SGSIM_STATUS_UNSUPPORTED = 5
} sgsim_status_t;

/**
 * @struct sgsim_t
 * @brief Structure to hold parameters and data for SGSIM simulation.
 *
 * This structure contains various parameters and data needed for performing
 * Sequential Gaussian Simulation (SGSIM).
 */
typedef struct {
    int x_len;  // Length of the realization in the x-direction.
    int realization_numbers;  // Number of realizetions to generate.
    int randomseed;
    int kriging_method;
    int if_alloc_memory;  // 1: library-owned output; 0: caller-owned output.
    int iteration_limit;  // The tolerance of maximum times of iteration error
    double* array;  // Array to store the simulated values.
    double z_min;  // Minimum simulated value.
    double z_max;  // Maximum simulated value.
    int constant_path;  // Reuse the first realization's random path.
} sgsim_t;

/**
 * Initialize a simulation request with explicit, scientifically neutral defaults.
 * The resulting bounds are unbounded and the kriging method is Simple Kriging.
 */
void sgsim_init_defaults(sgsim_t* sgsim);

/**
 * @brief Resolve legacy execution defaults for an sgsim_t structure.
 *
 * This function does not alter ``z_min`` or ``z_max``. Use
 * ``sgsim_init_defaults`` or set both bounds explicitly.
 * @param sgsim Pointer to an sgsim_t structure to set defaults for.
 * @param cov_model Pointer to the covariance model to be used.
 */
void set_sgsim_defaults(sgsim_t* sgsim, cov_model_t* cov_model);

/**
 * @brief Run Sequential Gaussian Simulation (SGSIM).
 *
 * This function performs Sequential Gaussian Simulation (SGSIM) based on the
 * provided sgsim_t parameters and covariance model.
 *
 * @param sgsim Pointer to the sgsim_t structure containing simulation parameters.
 * @param cov_model Pointer to the covariance model to be used.
 * @param vario_flag Flag indicating whether to calculate variogram or not.
 */
void sgsim_run(sgsim_t* sgsim, cov_model_t* cov_model, int vario_flag);

/**
 * Run the native 1D engine and return a machine-readable status code.
 * With ``if_alloc_memory == 1``, ``array`` must be NULL on entry and the
 * library allocates it. With ``if_alloc_memory == 0``, ``array`` must point to
 * caller-owned storage large enough for all output values.
 */
sgsim_status_t sgsim_run_checked(
    sgsim_t* sgsim,
    const cov_model_t* cov_model,
    int vario_flag);

/** Return a stable message for a native status code. */
const char* sgsim_status_message(sgsim_status_t status);

/**
 * @brief Free memory allocated for an sgsim_t structure.
 *
 * This function releases ``array`` only when ``if_alloc_memory == 1``. For
 * caller-owned storage it only clears the pointer in the structure.
 *
 * @param sgsim Pointer to the sgsim_t structure to free.
 */
void sgsim_t_free(sgsim_t* sgsim);

#endif   // UC_SGSIM_C_CORE_INCLUDE_SGSIM_H_
