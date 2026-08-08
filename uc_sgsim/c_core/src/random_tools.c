/**
 * @file random_tools.c
 * @brief Deterministic random helpers for the native SGSIM engine.
 *
 * Copyright (c) 2022-2023 Zncl2222
 * License: MIT
 */

# include <math.h>
# include <stdint.h>

# include "../include/random_tools.h"

# ifndef M_PI
# define M_PI 3.14159265358979323846
# endif

static uint32_t next_uint32(sgsim_rng_t* state) {
    return (uint32_t)mt19937_generate(&state->generator);
}

static uint32_t bounded_uint32(sgsim_rng_t* state, uint32_t bound) {
    uint64_t generator_range = UINT64_C(1) << 32;
    uint64_t acceptance_limit = generator_range - generator_range % bound;
    uint32_t value;
    do {
        value = next_uint32(state);
    } while ((uint64_t)value >= acceptance_limit);
    return value % bound;
}

void sgsim_rng_init(sgsim_rng_t* state, unsigned int seed) {
    mt19937_init(&state->generator, seed);
    state->has_spare_normal = 0;
    state->spare_normal = 0.0;
}

double sgsim_random_normal(sgsim_rng_t* state) {
    if (state->has_spare_normal) {
        state->has_spare_normal = 0;
        return state->spare_normal;
    }

    double first_uniform = ((double)next_uint32(state) + 1.0) / 4294967297.0;
    double second_uniform = (double)next_uint32(state) / 4294967296.0;
    double magnitude = sqrt(-2.0 * log(first_uniform));
    double angle = 2.0 * M_PI * second_uniform;

    state->spare_normal = magnitude * sin(angle);
    state->has_spare_normal = 1;
    return magnitude * cos(angle);
}

int* randompath(int* rpath, int length, sgsim_rng_t* rng_state) {
    for (int index = length - 1; index > 0; index--) {
        int random_index = (int)bounded_uint32(rng_state, (uint32_t)index + 1);
        int temporary = rpath[random_index];
        rpath[random_index] = rpath[index];
        rpath[index] = temporary;
    }
    return rpath;
}
