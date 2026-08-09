// Copyright 2022 Zncl2222


# include <stdio.h>
# include <math.h>
# include <stdint.h>
# include "../include/cov_model.h"
# include "../include/sgsim.h"
# include "../include/kriging.h"
# include "../include/matrix_tools.h"
# include "../include/native_array.h"
# include "../include/native_matrix.h"
# include "../include/variogram.h"
# include "utest.h"

UTEST(test, sgsim_simple_kriging) {
    sgsim_t sgsim_example;
    sgsim_init_defaults(&sgsim_example);
    sgsim_example.x_len = 150;
    sgsim_example.realization_numbers = 50;
    sgsim_example.randomseed = 12345;
    sgsim_example.if_alloc_memory = 1;

    cov_model_t cov_example = {
        .bw_l = 35,
        .bw_s = 1,
        .k_range = 17.32,
        .sill = 1,
        .nugget = 0,
        .max_neighbor = 8,
        .kind = COV_MODEL_GAUSSIAN,
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

UTEST(test, unconditional_ordinary_kriging_is_rejected) {
    sgsim_t sgsim_example = {
        .x_len = 75,
        .realization_numbers = 1,
        .randomseed = 15,
        .kriging_method = SGSIM_KRIGING_ORDINARY,
        .if_alloc_memory = 1,
        .z_min = -INFINITY,
        .z_max = INFINITY,
    };
    cov_model_t cov_example = {
        .bw_l = 35,
        .bw_s = 1,
        .k_range = 25,
        .sill = 1,
        .nugget = 0,
        .max_neighbor = 6,
        .kind = COV_MODEL_GAUSSIAN,
    };

    EXPECT_EQ(
        SGSIM_STATUS_UNSUPPORTED,
        sgsim_run_checked(&sgsim_example, &cov_example, 0));
    sgsim_t_free(&sgsim_example);
}

UTEST(test, sgsim_max_iteration) {
    sgsim_t sgsim_example = {
        .x_len = 75,
        .realization_numbers = 50,
        .randomseed = 15,
        .kriging_method = SGSIM_KRIGING_SIMPLE,
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

    sgsim_t_free(&simulation);
    EXPECT_TRUE(simulation.array == NULL);
    EXPECT_TRUE(isfinite(first[0]));
    EXPECT_TRUE(isfinite(second[0]));
}

UTEST(test, cholesky_solver_matches_two_neighbor_solution) {
    double values[] = {2.0, 0.0, 0.0, 0.0, -1.0};
    cov_model_t model = {
        .bw_l = 5,
        .bw_s = 1,
        .max_neighbor = 2,
        .k_range = 4.0,
        .sill = 1.7,
        .nugget = 0.2,
        .kind = COV_MODEL_EXPONENTIAL,
    };
    sampling_state sampling;
    kriging_workspace_t workspace;
    sgsim_rng_t rng_state;
    EXPECT_TRUE(sampling_state_init(&sampling, 5));
    EXPECT_TRUE(kriging_workspace_init(&workspace, 5, &model));

    sampling.sampled.data[0] = 0;
    sampling.sampled.data[1] = 4;
    sampling.currlen = 2;
    sampling.neighbor = 2;
    sampling_state_update(&sampling, 2);
    sgsim_rng_init(&rng_state, 2026);

    EXPECT_EQ(
        0,
        simple_kriging(
            values,
            &sampling,
            &workspace,
            &rng_state,
            0));

    double target_covariance = cov_model_at_lag(2.0, &model);
    double sample_covariance = cov_model_at_lag(4.0, &model);
    double expected_weight = target_covariance / (model.sill + sample_covariance);
    double expected_variance =
        model.sill - 2.0 * expected_weight * target_covariance;
    EXPECT_NEAR(expected_weight, workspace.weights.data[0], 1e-12);
    EXPECT_NEAR(expected_weight, workspace.weights.data[1], 1e-12);
    EXPECT_NEAR(sqrt(expected_variance), workspace.kriging_std, 1e-12);
    EXPECT_EQ(0.0, workspace.diagonal_jitter);

    sampling_state_free(&sampling);
    kriging_workspace_free(&workspace);
}

UTEST(test, cholesky_solver_regularizes_singular_covariance) {
    double values[] = {1.0, -1.0, 0.0};
    cov_model_t model = {
        .bw_l = 3,
        .bw_s = 1,
        .max_neighbor = 2,
        .k_range = 1e12,
        .sill = 1.0,
        .kind = COV_MODEL_GAUSSIAN,
    };
    sampling_state sampling;
    kriging_workspace_t workspace;
    sgsim_rng_t rng_state;
    EXPECT_TRUE(sampling_state_init(&sampling, 3));
    EXPECT_TRUE(kriging_workspace_init(&workspace, 3, &model));

    sampling.sampled.data[0] = 0;
    sampling.sampled.data[1] = 1;
    sampling.currlen = 2;
    sampling.neighbor = 2;
    sampling_state_update(&sampling, 2);
    sgsim_rng_init(&rng_state, 2026);

    EXPECT_EQ(
        0,
        simple_kriging(
            values,
            &sampling,
            &workspace,
            &rng_state,
            0));
    EXPECT_TRUE(workspace.diagonal_jitter > 0.0);
    EXPECT_TRUE(isfinite(values[2]));

    double explained_variance = 0.0;
    double weighted_covariance = 0.0;
    for (int row = 0; row < 2; row++) {
        explained_variance += workspace.weights.data[row]
            * workspace.covariance_vector.data[row];
        for (int column = 0; column < 2; column++) {
            weighted_covariance += workspace.weights.data[row]
                * workspace.covariance_matrix.data[
                    (size_t)row * workspace.covariance_matrix.stride
                    + (size_t)column]
                * workspace.weights.data[column];
        }
    }
    double expected_variance = model.sill
        - 2.0 * explained_variance + weighted_covariance;
    EXPECT_NEAR(expected_variance, workspace.kriging_std * workspace.kriging_std, 1e-12);

    sampling_state_free(&sampling);
    kriging_workspace_free(&workspace);
}

UTEST(test, zero_neighbors_generate_independent_draws) {
    const int realization_count = 12000;
    const int node_count = 4;
    sgsim_t simulation;
    sgsim_init_defaults(&simulation);
    simulation.x_len = node_count;
    simulation.realization_numbers = realization_count;
    simulation.randomseed = 20260808;
    simulation.if_alloc_memory = 1;

    cov_model_t model = {
        .bw_l = 10,
        .bw_s = 1,
        .max_neighbor = 0,
        .k_range = 10.0,
        .sill = 1.7,
        .nugget = 0.2,
        .kind = COV_MODEL_EXPONENTIAL,
    };

    EXPECT_EQ(SGSIM_STATUS_OK, sgsim_run_checked(&simulation, &model, 0));
    for (int left = 0; left < node_count; left++) {
        double left_mean = 0.0;
        for (int realization = 0; realization < realization_count; realization++) {
            left_mean += simulation.array[left + node_count * realization];
        }
        left_mean /= realization_count;
        EXPECT_NEAR(0.0, left_mean, 0.06);

        for (int right = left + 1; right < node_count; right++) {
            double right_mean = 0.0;
            for (int realization = 0; realization < realization_count; realization++) {
                right_mean += simulation.array[right + node_count * realization];
            }
            right_mean /= realization_count;

            double covariance = 0.0;
            for (int realization = 0; realization < realization_count; realization++) {
                covariance +=
                    (simulation.array[left + node_count * realization] - left_mean)
                    * (simulation.array[right + node_count * realization] - right_mean);
            }
            covariance /= realization_count - 1;
            EXPECT_NEAR(0.0, covariance, 0.08);
        }
    }
    sgsim_t_free(&simulation);
}

UTEST(test, explicit_defaults_and_zero_bounds_are_preserved) {
    sgsim_t defaults;
    sgsim_init_defaults(&defaults);
    EXPECT_TRUE(isinf(defaults.z_min) && defaults.z_min < 0.0);
    EXPECT_TRUE(isinf(defaults.z_max) && defaults.z_max > 0.0);
    EXPECT_EQ(SGSIM_KRIGING_SIMPLE, defaults.kriging_method);
    EXPECT_EQ(10, defaults.iteration_limit);

    sgsim_t lower_bounded = defaults;
    double output[64];
    lower_bounded.x_len = 1;
    lower_bounded.realization_numbers = 64;
    lower_bounded.randomseed = 7;
    lower_bounded.array = output;
    lower_bounded.z_min = 0.0;
    lower_bounded.iteration_limit = 100;

    cov_model_t model = {
        .bw_l = 1,
        .bw_s = 1,
        .max_neighbor = 0,
        .k_range = 1.0,
        .sill = 1.0,
        .kind = COV_MODEL_GAUSSIAN,
    };
    EXPECT_EQ(SGSIM_STATUS_OK, sgsim_run_checked(&lower_bounded, &model, 0));
    EXPECT_EQ(0.0, lower_bounded.z_min);
    EXPECT_TRUE(isinf(lower_bounded.z_max));
    EXPECT_EQ(0, model.max_neighbor);
    for (int index = 0; index < 64; index++) {
        EXPECT_GT(output[index], 0.0);
    }

    sgsim_t upper_bounded = defaults;
    upper_bounded.x_len = 1;
    upper_bounded.realization_numbers = 64;
    upper_bounded.randomseed = 11;
    upper_bounded.array = output;
    upper_bounded.z_max = 0.0;
    upper_bounded.iteration_limit = 100;
    EXPECT_EQ(SGSIM_STATUS_OK, sgsim_run_checked(&upper_bounded, &model, 0));
    EXPECT_TRUE(isinf(upper_bounded.z_min));
    EXPECT_EQ(0.0, upper_bounded.z_max);
    for (int index = 0; index < 64; index++) {
        EXPECT_LT(output[index], 0.0);
    }
}

UTEST(test, random_path_can_retain_or_swap_last_position) {
    int retained = 0;
    int swapped = 0;
    for (unsigned int seed = 0; seed < 128; seed++) {
        int path[] = {0, 1};
        sgsim_rng_t rng_state;
        sgsim_rng_init(&rng_state, seed);
        randompath(path, 2, &rng_state);
        retained += path[0] == 0;
        swapped += path[0] == 1;
    }
    EXPECT_GT(retained, 0);
    EXPECT_GT(swapped, 0);
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

    simulation.if_alloc_memory = 1;
    EXPECT_EQ(SGSIM_STATUS_INVALID_ARGUMENT, sgsim_run_checked(&simulation, &model, 0));
    simulation.if_alloc_memory = 0;
    simulation.array = NULL;
    EXPECT_EQ(SGSIM_STATUS_INVALID_ARGUMENT, sgsim_run_checked(&simulation, &model, 0));

    simulation.array = output;
    model.k_range = NAN;
    EXPECT_EQ(SGSIM_STATUS_INVALID_ARGUMENT, sgsim_run_checked(&simulation, &model, 0));
    model.k_range = 3.0;
    simulation.randomseed = -1;
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

UTEST(test, native_arrays_check_sizes_bounds_and_ownership) {
    sgsim_int_array_t integers = {0};
    sgsim_double_array_t doubles = {0};

    EXPECT_EQ(SGSIM_NUMERIC_OK, sgsim_int_array_init(&integers, 4));
    EXPECT_EQ(SGSIM_NUMERIC_OK, sgsim_int_array_iota(&integers));
    int integer_value = -1;
    EXPECT_EQ(SGSIM_NUMERIC_OK, sgsim_int_array_get(&integers, 3, &integer_value));
    EXPECT_EQ(3, integer_value);
    EXPECT_EQ(
        SGSIM_NUMERIC_INVALID_ARGUMENT,
        sgsim_int_array_get(&integers, 4, &integer_value));

    EXPECT_EQ(SGSIM_NUMERIC_OK, sgsim_double_array_init(&doubles, 2));
    EXPECT_EQ(SGSIM_NUMERIC_OK, sgsim_double_array_set(&doubles, 1, 2.5));
    double double_value = 0.0;
    EXPECT_EQ(
        SGSIM_NUMERIC_OK,
        sgsim_double_array_get(&doubles, 1, &double_value));
    EXPECT_EQ(2.5, double_value);
    EXPECT_EQ(
        SGSIM_NUMERIC_INVALID_ARGUMENT,
        sgsim_double_array_set(&doubles, 2, 1.0));

    sgsim_int_array_free(&integers);
    sgsim_int_array_free(&integers);
    sgsim_double_array_free(&doubles);
    sgsim_double_array_free(&doubles);
    EXPECT_TRUE(integers.data == NULL);
    EXPECT_EQ(0U, integers.length);

    EXPECT_EQ(
        SGSIM_NUMERIC_SIZE_OVERFLOW,
        sgsim_double_array_init(&doubles, SIZE_MAX));
    EXPECT_TRUE(doubles.data == NULL);
}

UTEST(test, native_matrix_cholesky_solves_spd_system) {
    sgsim_double_matrix_t matrix = {0};
    sgsim_double_matrix_t factor = {0};
    EXPECT_EQ(SGSIM_NUMERIC_OK, sgsim_double_matrix_init(&matrix, 3, 3));
    EXPECT_EQ(SGSIM_NUMERIC_OK, sgsim_double_matrix_init(&factor, 3, 3));

    const double values[] = {
        4.0, 1.0, 1.0,
        1.0, 3.0, 0.0,
        1.0, 0.0, 2.0,
    };
    for (size_t row = 0; row < 3; row++) {
        for (size_t column = 0; column < 3; column++) {
            EXPECT_EQ(
                SGSIM_NUMERIC_OK,
                sgsim_double_matrix_set(
                    &matrix,
                    row,
                    column,
                    values[row * 3 + column]));
        }
    }

    double jitter = -1.0;
    EXPECT_EQ(
        SGSIM_NUMERIC_OK,
        sgsim_cholesky_factorize_regularized(&matrix, &factor, 3, &jitter));
    EXPECT_EQ(0.0, jitter);

    double right_hand_side[] = {9.0, 7.0, 7.0};
    double work[3];
    double solution[3];
    EXPECT_EQ(
        SGSIM_NUMERIC_OK,
        sgsim_cholesky_solve(
            &factor,
            right_hand_side,
            work,
            solution,
            3));
    EXPECT_NEAR(1.0, solution[0], 1e-12);
    EXPECT_NEAR(2.0, solution[1], 1e-12);
    EXPECT_NEAR(3.0, solution[2], 1e-12);

    double quadratic_form = 0.0;
    EXPECT_EQ(
        SGSIM_NUMERIC_OK,
        sgsim_symmetric_quadratic_form(
            &matrix,
            solution,
            3,
            &quadratic_form));
    EXPECT_NEAR(44.0, quadratic_form, 1e-12);

    double value = 0.0;
    EXPECT_EQ(
        SGSIM_NUMERIC_INVALID_ARGUMENT,
        sgsim_double_matrix_get(&matrix, 3, 0, &value));
    sgsim_double_matrix_free(&matrix);
    sgsim_double_matrix_free(&factor);

    EXPECT_EQ(
        SGSIM_NUMERIC_SIZE_OVERFLOW,
        sgsim_double_matrix_init(&matrix, SIZE_MAX, 2));
    EXPECT_EQ(
        SGSIM_NUMERIC_SIZE_OVERFLOW,
        sgsim_double_matrix_init(
            &matrix,
            SIZE_MAX / sizeof(double) + 1U,
            1));
}

UTEST(test, native_matrix_rejects_nonfinite_input) {
    sgsim_double_matrix_t matrix = {0};
    sgsim_double_matrix_t factor = {0};
    EXPECT_EQ(SGSIM_NUMERIC_OK, sgsim_double_matrix_init(&matrix, 2, 2));
    EXPECT_EQ(SGSIM_NUMERIC_OK, sgsim_double_matrix_init(&factor, 2, 2));
    matrix.data[0] = 1.0;
    matrix.data[1] = NAN;
    matrix.data[2] = NAN;
    matrix.data[3] = 1.0;
    double jitter = 0.0;
    EXPECT_EQ(
        SGSIM_NUMERIC_NONFINITE_VALUE,
        sgsim_cholesky_factorize_regularized(&matrix, &factor, 2, &jitter));
    sgsim_double_matrix_free(&matrix);
    sgsim_double_matrix_free(&factor);
}

UTEST(test, legacy_lu_wrapper_solves_without_writing_past_result) {
    double row0[] = {4.0, 3.0};
    double row1[] = {6.0, 3.0};
    double* matrix[] = {row0, row1};
    double right_hand_side[] = {10.0, 12.0};
    double result[] = {0.0, 0.0, 1234.0};
    lu_inverse_solver(matrix, right_hand_side, result, 2);
    EXPECT_NEAR(1.0, result[0], 1e-12);
    EXPECT_NEAR(2.0, result[1], 1e-12);
    EXPECT_EQ(1234.0, result[2]);
}

UTEST(test, variogram_uses_lag_windows_without_quadratic_storage) {
    double values[] = {0.0, 1.0, 2.0, 3.0};
    double result[] = {-1.0, -1.0, -1.0, -1.0};
    variogram(values, result, 4, 4, 1);
    EXPECT_NEAR(0.5, result[0], 1e-12);
    EXPECT_NEAR(11.0 / 10.0, result[1], 1e-12);
    EXPECT_NEAR(5.0 / 3.0, result[2], 1e-12);
    EXPECT_NEAR(17.0 / 6.0, result[3], 1e-12);

    double shifted[] = {1e12 + 1.0, 1e12 + 2.0, 1e12 + 3.0};
    EXPECT_NEAR(2.0 / 3.0, variance(shifted, 3), 1e-12);
    EXPECT_TRUE(isnan(variance(NULL, 0)));
}

UTEST_MAIN();
