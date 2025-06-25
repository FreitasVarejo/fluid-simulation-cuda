// src/main_serial.c

#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include "solver.h"
#include "helpers.h"

int main() {
    double u[NX][NY], v[NX][NY], p[NX][NY];
    initialize(u, v, p);

    const char *variant = getenv_or("WAVE_VARIANT", "serial");   // ou "openmp"/"cuda" no respectivo main
    const char *cfg     = getenv_or("WAVE_CFG",     "N100_T32_TB1");
    const char *param   = getenv_or("WAVE_PARAM",   "-");

    double start = omp_get_wtime();
    for (int t = 0; t < NT; ++t) {
        solve_pressure(u, v, p);
        update_velocities(u, v, p);
        if (t % 10 == 0) save_results(variant, cfg, param, t / 10, p);
    }
    printf("serial: %.6f\n", omp_get_wtime() - start);
    return 0;
}
