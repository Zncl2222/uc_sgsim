// Copyright 2022 Zncl2222


# include <stdio.h>
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

    sgsim_run(&sgsim_example, &cov_example, 0);
    sgsim_run(&sgsim_example, &cov_example, 1);
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

    sgsim_run(&sgsim_example, &cov_example, 0);
    sgsim_run(&sgsim_example, &cov_example, 1);
    sgsim_t_free(&sgsim_example);
}

UTEST(test, sgsim_simple_kriging_with_cov_cache) {
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
        .use_cov_cache = 1,
        .sill = 1,
        .nugget = 0,
        .max_neighbor = 9
    };

    sgsim_run(&sgsim_example, &cov_example, 0);
    sgsim_run(&sgsim_example, &cov_example, 1);
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

    sgsim_run(&sgsim_example, &cov_example, 0);
    sgsim_run(&sgsim_example, &cov_example, 1);
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

    sgsim_run(&sgsim_example, &cov_example, 0);
    sgsim_run(&sgsim_example, &cov_example, 1);
    sgsim_t_free(&sgsim_example);
}

UTEST(test, sgsim_ordinary_kriging_with_cov_cache) {
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
        .use_cov_cache = 1,
        .sill = 1,
        .nugget = 0,
        .max_neighbor = 6
    };

    sgsim_run(&sgsim_example, &cov_example, 0);
    sgsim_run(&sgsim_example, &cov_example, 1);
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
        .use_cov_cache = 1,
        .sill = 1,
        .nugget = 0,
        .max_neighbor = 6
    };

    sgsim_run(&sgsim_example, &cov_example, 0);
    sgsim_t_free(&sgsim_example);
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
