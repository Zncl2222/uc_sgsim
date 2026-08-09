// Copyright 2022 Zncl2222

# include <stdio.h>
# include <stdlib.h>

# include "./uc_sgsim/c_core/include/sgsim.h"
# include "./uc_sgsim/c_core/include/cov_model.h"
# if defined(__linux__) || defined(__unix__)
# define PAUSE printf("Press Enter key to continue..."); fgetc(stdin);//NOLINT
# elif _WIN32
# define PAUSE system("PAUSE");
# endif

int main() {
    // Initialize explicit unbounded defaults, then configure this run.
    sgsim_t sgsim_example;
    sgsim_init_defaults(&sgsim_example);
    sgsim_example.x_len = 150;
    sgsim_example.realization_numbers = 5;
    sgsim_example.randomseed = 12345;
    sgsim_example.if_alloc_memory = 1;

    // max_neighbor=0 is valid and means independent draws; choose 8 here.
    cov_model_t cov_example = {
        .bw_l = 35,
        .bw_s = 1,
        .k_range = 17.32,
        .use_cov_cache = 0,
        .sill = 1,
        .nugget = 0,
        .max_neighbor = 8,
        .kind = COV_MODEL_GAUSSIAN,
    };

    sgsim_status_t status = sgsim_run_checked(&sgsim_example, &cov_example, 0);
    if (status != SGSIM_STATUS_OK) {
        fprintf(stderr, "%s\n", sgsim_status_message(status));
        sgsim_t_free(&sgsim_example);
        return 1;
    }
    sgsim_t_free(&sgsim_example);
    PAUSE
    return 0;
}
