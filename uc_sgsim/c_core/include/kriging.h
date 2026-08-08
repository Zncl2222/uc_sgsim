/**
 * @file kriging.h
 * @brief Header file for Kriging functions in the SGSIM library.
 *
 * This header defines the data structures and functions related to Kriging
 * used in the Sequential Gaussian Simulation (SGSIM) library.
 *
 * Copyright (c) 2022-2023 Zncl2222
 * License: MIT
 */

#ifndef UC_SGSIM_C_CORE_INCLUDE_KRIGING_H_
#define UC_SGSIM_C_CORE_INCLUDE_KRIGING_H_

# include "../include/cov_model.h"
# include "../include/random_tools.h"

/**
 * @struct sampling_state
 * @brief Structure to represent the state of sampling during Kriging.
 *
 * This structure maintains the state of sampling during the Kriging process,
 * including information about neighbors, current length, and sampled points.
 */
typedef struct {
    int neighbor;           // Number of neighbors considered.
    int currlen;            // Current length of sampled points.
    int unsampled_point;    // Grid index currently being simulated.
    int* sampled;           // Previously sampled grid indices.
} sampling_state;

/** One candidate conditioning point. */
typedef struct {
    int grid_index;
    double value;
    double distance;
} kriging_neighbor_t;

/** Contiguous, per-simulation scratch buffers used by kriging. */
typedef struct {
    const cov_model_t* model;
    int matrix_stride;
    kriging_neighbor_t* candidates;
    double* covariance_vector;
    double* covariance_matrix;
    double* factor_matrix;
    double* weights;
    double* solve_temp;
    double* unit_solution;
    double kriging_std;
    double diagonal_jitter;
} kriging_workspace_t;

/**
 * @brief Initialize a sampling state structure.
 *
 * This function initializes a sampling_state structure with default values.
 *
 * @param sampling Pointer to the sampling_state structure to initialize.
 * @param x_grid_len Length of the grid used in sampling.
 */
int sampling_state_init(sampling_state* sampling, int x_grid_len);

/** Release buffers owned by a sampling state. */
void sampling_state_free(sampling_state* sampling);

/**
 * @brief Update the sampling state with a new unsampled point.
 *
 * This function updates the sampling state with a new unsampled point and
 * its index in the grid.
 *
 * @param sampling Pointer to the sampling_state structure to update.
 * @param unsampled_point The unsampled point to add.
 */
void sampling_state_update(sampling_state* sampling, int unsampled_point);

/**
 * @brief Set Kriging parameters for the simulation.
 *
 * This function sets the Kriging parameters based on the length of the grid
 * and the covariance model to be used.
 *
 * @param x_len Length of the grid.
 * @param cov_model Pointer to the covariance model.
 */
int kriging_workspace_init(
    kriging_workspace_t* workspace,
    int x_len,
    const cov_model_t* cov_model);

/** Release all scratch buffers owned by a kriging workspace. */
void kriging_workspace_free(kriging_workspace_t* workspace);

/**
 * @brief Perform simple Kriging to estimate values at unsampled points.
 *
 * This function performs simple Kriging to estimate values at unsampled points
 * based on the sampling state, random number generator state, and Kriging method.
 *
 * @param array Array to store estimated values.
 * @param sampling Pointer to the sampling state.
 * @param rng_state Pointer to the random number generator state.
 * @param kriging_method Kriging method to use (e.g., ordinary kriging).
 * @param use_solution_cache Reuse weights and standard deviation loaded by the caller.
 */
int simple_kriging(
    double* array,
    sampling_state* sampling,
    kriging_workspace_t* workspace,
    sgsim_rng_t* rng_state,
    int kriging_method,
    int use_solution_cache);

#endif  // UC_SGSIM_C_CORE_INCLUDE_KRIGING_H_
