// Copyright 2022 Zncl2222


# include <stdio.h>
# include "../include/cov_model.h"
# include "utest.h"

UTEST(test, sgsim) {
    int mlen = 150;
    int nR = 5;
    int hs = 35;
    int bw = 1;
    int randomseed = 12345;

    sgsim(mlen, nR, hs, bw, 17.32, 1, randomseed, 0);
    sgsim(mlen, nR, hs, bw, 17.32, 1, randomseed, 1);
}


UTEST_MAIN();