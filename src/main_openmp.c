// src/main_openmp.c

#include <stdio.h>
#include <omp.h>
#include "solver.h"

int main() {
    double u[NX][NY], v[NX][NY], p[NX][NY];
    initialize(u,v,p);

    double start = omp_get_wtime();

    #pragma omp parallel
    {
        for (int t = 0; t < NT; ++t) {

            /* pressão */
            solve_pressure(u,v,p);   /* troque parallel/for interno por for */

            /* velocidades */
            update_velocities(u,v,p);

            #pragma omp master   /* só 1 thread faz I/O */
            if (t % 10 == 0) save_results(t/10, p);
        }
    }

    printf("openmp: %.6f\n", omp_get_wtime() - start);
}
