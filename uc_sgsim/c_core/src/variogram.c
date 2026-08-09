/**
 * @file variogram.c
 * @brief Implementation of Variogram Calculation
 *
 * This source file contains the implementation of variogram calculation functions. Variograms are
 * statistical measures used in geostatistics to quantify the spatial variability of a dataset.
 * The functions in this file calculate the experimental variogram and variance of a given dataset.
 *
 * Copyright (c) 2022-2023 Zncl2222
 * License: MIT
 */

# include <math.h>
# include <stddef.h>

# include "../include/variogram.h"

void variogram(const double* array, double* v, int mlen, int bw, int bw_s) {
    if (array == NULL || v == NULL || mlen <= 0 || bw <= 0 || bw_s <= 0) {
        return;
    }
    size_t value_count = (size_t)mlen;
    size_t lag_step = (size_t)bw_s;
    for (size_t bin = 0; bin < (size_t)bw; bin += lag_step) {
        size_t lower_lag = bin > lag_step ? bin - lag_step : 1U;
        size_t upper_lag = bin + lag_step;
        upper_lag = upper_lag >= value_count ? value_count - 1U : upper_lag;

        double squared_difference_sum = 0.0;
        double compensation = 0.0;
        size_t pair_count = 0;
        for (size_t lag = lower_lag; lag <= upper_lag; lag++) {
            for (size_t left = 0; left < value_count - lag; left++) {
                double difference = array[left + lag] - array[left];
                double term = difference * difference;
                double corrected = term - compensation;
                double updated = squared_difference_sum + corrected;
                compensation = (updated - squared_difference_sum) - corrected;
                squared_difference_sum = updated;
                pair_count++;
            }
        }

        if (squared_difference_sum >= 1e-6 && pair_count != 0) {
            v[bin] = squared_difference_sum / (2.0 * (double)pair_count);
        }
    }
}

double variance(const double* array, int mlen) {
    if (array == NULL || mlen <= 0) {
        return NAN;
    }
    double mean = 0;
    double sum_of_squares = 0;
    for (int i = 0; i < mlen; i++) {
        double delta = array[i] - mean;
        mean += delta / (double)(i + 1);
        double updated_delta = array[i] - mean;
        sum_of_squares += delta * updated_delta;
    }
    return sum_of_squares / (double)mlen;
}
