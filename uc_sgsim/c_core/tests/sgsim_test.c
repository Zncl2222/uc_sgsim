// Copyright 2022 Zncl2222


# include <stdio.h>
# include <math.h>
# include "../include/cov_model.h"
# include "../include/sgsim.h"
# include "../include/kriging.h"
# include "../include/variogram.h"
# include "utest.h"

UTEST(test, sgsim_simple_kriging) {
    sgsim_t sgsim_example = {
        .x_len = 150,
        .realization_numbers = 50,
        .randomseed = 12345,
        .kriging_method = 0,
        .if_alloc_memory = 1,
    };

    cov_model_t cov_example = {
        .bw_l = 35,
        .bw_s = 1,
        .k_range = 17.32,
        .sill = 1,
        .nugget = 0,
    };

    EXPECT_EQ(SGSIM_STATUS_OK, sgsim_run_checked(&sgsim_example, &cov_example, 0));
    sgsim_t_free(&sgsim_example);
}

UTEST(test, sgsim_simple_kriging2) {
    sgsim_t sgsim_example = {
        .x_len = 300,
        .realization_numbers = 16,
        .randomseed = 987,
        .kriging_method = 0,
        .if_alloc_memory = 1,
        .z_min = -6,
        .z_max = 6
    };
    cov_model_t cov_example = {
        .bw_l = 35,
        .bw_s = 1,
        .k_range = 25,
        .sill = 1,
        .nugget = 0,
        .max_neighbor = 9
    };

    EXPECT_EQ(SGSIM_STATUS_OK, sgsim_run_checked(&sgsim_example, &cov_example, 0));
    sgsim_t_free(&sgsim_example);
}

UTEST(test, sgsim_simple_kriging_with_cov_cache) {
    sgsim_t sgsim_example = {
        .x_len = 300,
        .realization_numbers = 16,
        .randomseed = 987,
        .kriging_method = 0,
        .if_alloc_memory = 1,
        .constant_path = 1,
        .z_min = -6,
        .z_max = 6
    };
    cov_model_t cov_example = {
        .bw_l = 35,
        .bw_s = 1,
        .k_range = 25,
        .use_cov_cache = 1,
        .sill = 1,
        .nugget = 0,
        .max_neighbor = 9
    };

    EXPECT_EQ(SGSIM_STATUS_OK, sgsim_run_checked(&sgsim_example, &cov_example, 0));
    sgsim_t_free(&sgsim_example);
}

UTEST(test, sgsim_ordinary_kriging) {
    sgsim_t sgsim_example = {
        .x_len = 150,
        .realization_numbers = 50,
        .randomseed = 15,
        .kriging_method = 1,
        .if_alloc_memory = 1,
    };
    cov_model_t cov_example = {
        .bw_l = 35,
        .bw_s = 1,
        .k_range = 25,
        .sill = 1,
        .nugget = 0,
    };

    EXPECT_EQ(SGSIM_STATUS_OK, sgsim_run_checked(&sgsim_example, &cov_example, 0));
    sgsim_t_free(&sgsim_example);
}

UTEST(test, sgsim_ordinary_kriging2) {
    sgsim_t sgsim_example = {
        .x_len = 75,
        .realization_numbers = 50,
        .randomseed = 15,
        .kriging_method = 1,
        .if_alloc_memory = 1,
        .z_min = -3,
        .z_max = 3
    };
    cov_model_t cov_example = {
        .bw_l = 35,
        .bw_s = 1,
        .k_range = 12,
        .sill = 1,
        .nugget = 0,
        .max_neighbor = 6
    };

    EXPECT_EQ(SGSIM_STATUS_OK, sgsim_run_checked(&sgsim_example, &cov_example, 0));
    sgsim_t_free(&sgsim_example);
}

UTEST(test, sgsim_ordinary_kriging_with_cov_cache) {
    sgsim_t sgsim_example = {
        .x_len = 75,
        .realization_numbers = 50,
        .randomseed = 15,
        .kriging_method = 1,
        .if_alloc_memory = 1,
        .constant_path = 1,
        .z_min = -3,
        .z_max = 3
    };
    cov_model_t cov_example = {
        .bw_l = 35,
        .bw_s = 1,
        .k_range = 12,
        .use_cov_cache = 1,
        .sill = 1,
        .nugget = 0,
        .max_neighbor = 6
    };

    EXPECT_EQ(SGSIM_STATUS_OK, sgsim_run_checked(&sgsim_example, &cov_example, 0));
    sgsim_t_free(&sgsim_example);
}

UTEST(test, sgsim_max_iteration) {
    sgsim_t sgsim_example = {
        .x_len = 75,
        .realization_numbers = 50,
        .randomseed = 15,
        .kriging_method = 1,
        .if_alloc_memory = 1,
        .iteration_limit = 3,
        .z_min = -1,
        .z_max = 1
    };
    cov_model_t cov_example = {
        .bw_l = 35,
        .bw_s = 1,
        .k_range = 12,
        .use_cov_cache = 0,
        .sill = 1,
        .nugget = 0,
        .max_neighbor = 6
    };

    EXPECT_EQ(
        SGSIM_STATUS_ITERATION_LIMIT,
        sgsim_run_checked(&sgsim_example, &cov_example, 0));
    sgsim_t_free(&sgsim_example);
}

UTEST(test, covariance_models_match_contract) {
    double lags[] = {0.0, 3.0};
    double covariance[2];
    cov_model_t model = {
        .bw_l = 10,
        .bw_s = 1,
        .k_range = 3.0,
        .sill = 1.0,
        .nugget = 0.2,
    };

    model.kind = COV_MODEL_GAUSSIAN;
    cov_compute(lags, covariance, 2, &model);
    EXPECT_NEAR(1.0, covariance[0], 1e-12);
    EXPECT_NEAR(0.8 * exp(-3.0), covariance[1], 1e-12);

    model.kind = COV_MODEL_EXPONENTIAL;
    cov_compute(lags, covariance, 2, &model);
    EXPECT_NEAR(1.0, covariance[0], 1e-12);
    EXPECT_NEAR(0.8 * exp(-3.0), covariance[1], 1e-12);

    model.kind = COV_MODEL_SPHERICAL;
    cov_compute(lags, covariance, 2, &model);
    EXPECT_NEAR(1.0, covariance[0], 1e-12);
    EXPECT_NEAR(0.0, covariance[1], 1e-12);
}

UTEST(test, caller_owned_output_is_deterministic) {
    double first[48];
    double second[48];
    sgsim_t simulation = {
        .x_len = 12,
        .realization_numbers = 4,
        .randomseed = 2026,
        .kriging_method = SGSIM_KRIGING_SIMPLE,
        .if_alloc_memory = 0,
        .iteration_limit = 10,
        .array = first,
        .z_min = -INFINITY,
        .z_max = INFINITY,
    };
    cov_model_t model = {
        .bw_l = 10,
        .bw_s = 1,
        .max_neighbor = 8,
        .k_range = 4.0,
        .sill = 1.7,
        .nugget = 0.2,
        .kind = COV_MODEL_EXPONENTIAL,
    };

    EXPECT_EQ(SGSIM_STATUS_OK, sgsim_run_checked(&simulation, &model, 0));
    simulation.array = second;
    EXPECT_EQ(SGSIM_STATUS_OK, sgsim_run_checked(&simulation, &model, 0));

    for (int i = 0; i < 48; i++) {
        EXPECT_TRUE(isfinite(first[i]));
        EXPECT_EQ(first[i], second[i]);
    }
}

UTEST(test, unsupported_and_invalid_options_return_status) {
    double output[4];
    sgsim_t simulation = {
        .x_len = 4,
        .realization_numbers = 1,
        .randomseed = 7,
        .kriging_method = SGSIM_KRIGING_SIMPLE,
        .array = output,
        .z_min = -INFINITY,
        .z_max = INFINITY,
    };
    cov_model_t model = {
        .bw_l = 4,
        .bw_s = 1,
        .use_cov_cache = 1,
        .k_range = 3.0,
        .sill = 1.0,
        .kind = COV_MODEL_GAUSSIAN,
    };

    EXPECT_EQ(SGSIM_STATUS_INVALID_ARGUMENT, sgsim_run_checked(&simulation, &model, 0));
    model.use_cov_cache = 0;
    EXPECT_EQ(SGSIM_STATUS_UNSUPPORTED, sgsim_run_checked(&simulation, &model, 1));
    model.k_range = NAN;
    EXPECT_EQ(SGSIM_STATUS_INVALID_ARGUMENT, sgsim_run_checked(&simulation, &model, 0));
}

UTEST(test, variance) {
    double arr[20];
    for (int i = 0; i < 20; i++) {
        arr[i] = i + i * 2;
    }

    double var = variance(arr, 20);
    EXPECT_NEAR(299.25, var, 1e-12);
}

UTEST_MAIN();
