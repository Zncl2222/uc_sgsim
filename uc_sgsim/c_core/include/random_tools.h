/**
 * @file random_tools.h
 * @brief Random Tools Library
 *
 * This header file provides declarations for functions related to random number generation and
 * random path generation. The functions in this library assist in generating random paths
 * or sequences of integers.
 *
 * Copyright (c) 2022-2023 Zncl2222
 * License: MIT
 */

#ifndef UC_SGSIM_C_CORE_INCLUDE_RANDOM_TOOLS_H_
#define UC_SGSIM_C_CORE_INCLUDE_RANDOM_TOOLS_H_

# include <stddef.h>
# include <stdint.h>

# define SGSIM_MT19937_STATE_SIZE 624

/** RNG state with a cached Box-Muller pair for the native simulator. */
typedef struct {
    uint32_t generator_state[SGSIM_MT19937_STATE_SIZE];
    size_t generator_index;
    int has_spare_normal;
    double spare_normal;
} sgsim_rng_t;

/** Initialize all native random streams from one deterministic seed. */
void sgsim_rng_init(sgsim_rng_t* state, unsigned int seed);

/** Draw a finite standard normal value, retaining the paired Box-Muller draw. */
double sgsim_random_normal(sgsim_rng_t* state);

/**
 * @brief Generate a random path of integers.
 *
 * This function generates a random path of integers of the specified length using the provided
 * Mersenne Twister random number generator state.
 *
 * @param rpath An array for generating the random path.
 * @param length The length of the random path to generate.
 * @param rng_state A pointer to the Mersenne Twister random number generator state.
 *
 * @return A pointer to the generated random path (same as `rpath`).
 */
int* randompath(int* rpath, int length, sgsim_rng_t* rng_state);

#endif  // UC_SGSIM_C_CORE_INCLUDE_RANDOM_TOOLS_H_
