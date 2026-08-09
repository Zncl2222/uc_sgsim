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

/*
 * Copyright 1996-2002 Makoto Matsumoto and Takuji Nishimura.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. The names of its contributors may not be used to endorse or promote
 *    products derived from this software without specific prior written
 *    permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

# define MT_M 397U
# define MT_MATRIX_A UINT32_C(0x9908b0df)
# define MT_UPPER_MASK UINT32_C(0x80000000)
# define MT_LOWER_MASK UINT32_C(0x7fffffff)

static void mt19937_seed(sgsim_rng_t* state, uint32_t seed) {
    state->generator_state[0] = seed;
    for (size_t index = 1; index < SGSIM_MT19937_STATE_SIZE; index++) {
        uint32_t previous = state->generator_state[index - 1];
        state->generator_state[index] = UINT32_C(1812433253)
            * (previous ^ (previous >> 30U)) + (uint32_t)index;
    }
    state->generator_index = SGSIM_MT19937_STATE_SIZE;
}

static void mt19937_twist(sgsim_rng_t* state) {
    for (size_t index = 0; index < SGSIM_MT19937_STATE_SIZE; index++) {
        uint32_t combined =
            (state->generator_state[index] & MT_UPPER_MASK)
            | (state->generator_state[(index + 1U) % SGSIM_MT19937_STATE_SIZE]
                & MT_LOWER_MASK);
        uint32_t twisted = combined >> 1U;
        if ((combined & UINT32_C(1)) != 0U) {
            twisted ^= MT_MATRIX_A;
        }
        state->generator_state[index] =
            state->generator_state[(index + MT_M) % SGSIM_MT19937_STATE_SIZE]
            ^ twisted;
    }
    state->generator_index = 0;
}

static uint32_t next_uint32(sgsim_rng_t* state) {
    if (state->generator_index >= SGSIM_MT19937_STATE_SIZE) {
        mt19937_twist(state);
    }
    uint32_t value = state->generator_state[state->generator_index++];
    value ^= value >> 11U;
    value ^= (value << 7U) & UINT32_C(0x9d2c5680);
    value ^= (value << 15U) & UINT32_C(0xefc60000);
    value ^= value >> 18U;
    return value;
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
    mt19937_seed(state, (uint32_t)seed);
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
