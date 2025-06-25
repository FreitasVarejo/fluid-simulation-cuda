// src/main_serial.c

#include <stdio.h>
#include <omp.h>
#include "solver.h"

int main() {
    double u[NX][NY], v[NX][NY], p[NX][NY];
    initialize(u, v, p);

    double start = omp_get_wtime();
    for (int t = 0; t < NT; ++t) {
        solve_pressure(u, v, p);
        update_velocities(u, v, p);
        if (t % 10 == 0) save_results(t / 10, p);
    }
    printf("serial: %.6f\n", omp_get_wtime() - start);
    return 0;
}
