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

void cov_compute(const double* x, double* cov, int n_dim, const cov_model_t* cov_model) {
    double partial_sill = cov_model->sill - cov_model->nugget;
    for (int i = 0; i < n_dim; i++) {
        double factor =
            (1 - exp(-3 * (x[i] * x[i]) / (cov_model->k_range * cov_model->k_range)));
        cov[i] = cov_model->sill - (partial_sill * factor) + cov_model->nugget;
    }
}

void cov_compute2d(double* const* x, double* cov, int n_dim, const cov_model_t* cov_model) {
    double partial_sill = cov_model->sill - cov_model->nugget;
    int index = 0;
    for (int i = 0; i < n_dim; i++) {
        for (int j = 0; j < n_dim; j++) {
            double factor =
                (1 - exp(-3 * (x[i][j] * x[i][j]) / (cov_model->k_range * cov_model->k_range)));
            cov[index] = cov_model->sill - (partial_sill * factor) + cov_model->nugget;
            index++;
        }
    }
}
