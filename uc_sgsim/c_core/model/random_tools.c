// Copyright 2022 Zncl2222

# include "..\header\random_tools.h"
# include <stdlib.h>
# include <stdio.h>
# include <math.h>

int* randompath(int* rpath, int length) {
    int rtemp;

    for (int i = length - 1; i; i--) {
        int rindex = rand() % i;

        do {
            rtemp = rpath[rindex];
            rpath[rindex] = rpath[i];
            rpath[i] = rtemp;
        }while(0);
    }

    return rpath;
}

float random_normal() {
    float x = rand() / (float)RAND_MAX;
    float y = rand() / (float)RAND_MAX;
    float z = sqrt(-2 * log(x)) * cos(2 * M_PI * y);

    return z;
}