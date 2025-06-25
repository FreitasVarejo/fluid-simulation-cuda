// src/main_openmp.c

#include <stdio.h>
#include <omp.h>
#include "solver.h"

int main() {
    double u[NX][NY], v[NX][NY], p[NX][NY];
    omp_set_num_threads(4);
    initialize(u, v, p);

    double start = omp_get_wtime();
    for (int t = 0; t < NT; ++t) {
        solve_pressure(u, v, p);
        update_velocities(u, v, p);
        if (t % 10 == 0) save_results(t / 10, p);
    }
    printf("openmp: %.6f\n", omp_get_wtime() - start);
    return 0;
}
