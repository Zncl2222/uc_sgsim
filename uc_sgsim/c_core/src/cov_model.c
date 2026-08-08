/**
 * @file cov_model.c
 * @brief Implementation of Covariance Models
 *
 * This source file contains the implementation of covariance model functions used in geostatistics.
 * These functions provide utilities for setting default covariance model parameters and calculating
 * covariance values based on specified models.
 *
 * Copyright (c) 2022-2023 Zncl2222
 * License: MIT
 */

# include <math.h>
# include "../include/cov_model.h"

void set_cov_model_default(cov_model_t* cov_model) {
    cov_model->max_neighbor = cov_model->max_neighbor == 0 ? 4 : cov_model->max_neighbor;
    cov_model->sill = cov_model->sill == 0 ? 1 : cov_model->sill;
    cov_model->bw = cov_model->bw_l / cov_model->bw_s;
}

static double covariance_at_lag(double lag, const cov_model_t* cov_model) {
    if (lag == 0.0) {
        return cov_model->sill;
    }

    double partial_sill = cov_model->sill - cov_model->nugget;
    double scaled_lag = lag / cov_model->k_range;

    switch (cov_model->kind) {
        case COV_MODEL_EXPONENTIAL:
            return partial_sill * exp(-3.0 * scaled_lag);
        case COV_MODEL_SPHERICAL:
            if (scaled_lag >= 1.0) {
                return 0.0;
            }
            return partial_sill * (1.0 - 1.5 * scaled_lag + 0.5 * pow(scaled_lag, 3.0));
        case COV_MODEL_GAUSSIAN:
        default:
            return partial_sill * exp(-3.0 * scaled_lag * scaled_lag);
    }
}

void cov_compute(const double* x, double* cov, int n_dim, const cov_model_t* cov_model) {
    for (int i = 0; i < n_dim; i++) {
        cov[i] = covariance_at_lag(x[i], cov_model);
    }
}

void cov_compute2d(double* const* x, double* cov, int n_dim, const cov_model_t* cov_model) {
    int index = 0;
    for (int i = 0; i < n_dim; i++) {
        for (int j = 0; j < n_dim; j++) {
            cov[index] = covariance_at_lag(x[i][j], cov_model);
            index++;
        }
    }
}
