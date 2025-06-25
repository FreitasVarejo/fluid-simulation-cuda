// src/solver.c

#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "solver.h"

void initialize(double u[NX][NY], double v[NX][NY], double p[NX][NY]) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            u[i][j] = 0.0;
            v[i][j] = 0.0;
            p[i][j] = 0.0;
            double dist = sqrt((i - NX/2.0)*(i - NX/2.0) + (j - NY/2.0)*(j - NY/2.0));
            if (dist < 10.0) {
                p[i][j] = 10.0 * exp(-dist*dist / 25.0);
            }
        }
    }
}


// Função para resolver a equação de Poisson para pressão
void solve_pressure(double u[NX][NY], double v[NX][NY], double p[NX][NY])
{
    static double p_new[NX][NY];
    const int    max_iter   = 1000;
    const double tolerance  = 1e-6;

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < NX; ++i)
        for (int j = 0; j < NY; ++j)
            p_new[i][j] = p[i][j];

    int    iter;
    double max_error;

    // laço de relaxação
    #pragma omp parallel
    {
        for (iter = 0; iter < max_iter; ++iter) {

            double local_max = 0.0;
            #pragma omp for collapse(2) nowait
            for (int i = 1; i < NX - 1; ++i) {
                for (int j = 1; j < NY - 1; ++j) {

                    const double div_u =
                          (u[i+1][j] - u[i-1][j]) * (0.5 / DX)
                        + (v[i][j+1] - v[i][j-1]) * (0.5 / DY);

                    const double p_old = p_new[i][j];

                    p_new[i][j] = ( (p_new[i+1][j] + p_new[i-1][j]) * DY*DY +
                                    (p_new[i][j+1] + p_new[i][j-1]) * DX*DX -
                                    RHO * DX*DX * DY*DY * div_u / DT ) /
                                   (2.0 * (DX*DX + DY*DY));

                    const double err = fabs(p_new[i][j] - p_old);
                    if (err > local_max) local_max = err;
                }
            }

            #pragma omp critical
            {
                if (local_max > max_error) max_error = local_max;
            }
            #pragma omp barrier

            if (max_error <= tolerance) break;
            max_error = 0.0;

            #pragma omp single
            {
                for (int i = 0; i < NX; ++i) {
                    p_new[i][0]    = p_new[i][1];
                    p_new[i][NY-1] = p_new[i][NY-2];
                }
                for (int j = 0; j < NY; ++j) {
                    p_new[0][j]    = p_new[1][j];
                    p_new[NX-1][j] = p_new[NX-2][j];
                }
            }
            #pragma omp barrier
        }
    }

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < NX; ++i)
        for (int j = 0; j < NY; ++j)
            p[i][j] = p_new[i][j];
}

void update_velocities(double u[NX][NY],
                       double v[NX][NY],
                       double p[NX][NY])
{
    /* buffers alinhados para ajudar o vetor de 512 bits das AVX-512 */
    static double u_new[NX][NY] __attribute__((aligned(64)));
    static double v_new[NX][NY] __attribute__((aligned(64)));

    /* ───── varredura em blocos (tiling espacial) ─────────────────── */
    for (int ii = 1; ii < NX-1; ii += TILE) {
        for (int jj = 1; jj < NY-1; jj += TILE) {

            const int i_max = (ii + TILE < NX-1) ? ii + TILE : NX-1;
            const int j_max = (jj + TILE < NY-1) ? jj + TILE : NY-1;

            /* paralelismo entre linhas ................................ */
            #pragma omp parallel for schedule(static)
            for (int i = ii; i < i_max; ++i) {

                /* vetorização sobre colunas ........................... */
                #pragma omp simd
                for (int j = jj; j < j_max; ++j) {

                    /* convection */
                    const double u_conv =  u[i][j]*(u[i+1][j] - u[i-1][j])*(0.5 / DX)
                                         + v[i][j]*(u[i][j+1] - u[i][j-1])*(0.5 / DY);

                    const double v_conv =  u[i][j]*(v[i+1][j] - v[i-1][j])*(0.5 / DX)
                                         + v[i][j]*(v[i][j+1] - v[i][j-1])*(0.5 / DY);

                    /* diffusion */
                    const double u_diff = NU * ( (u[i+1][j]-2*u[i][j]+u[i-1][j])/(DX*DX)
                                               + (u[i][j+1]-2*u[i][j]+u[i][j-1])/(DY*DY) );

                    const double v_diff = NU * ( (v[i+1][j]-2*v[i][j]+v[i-1][j])/(DX*DX)
                                               + (v[i][j+1]-2*v[i][j]+v[i][j-1])/(DY*DY) );

                    /* pressure */
                    const double u_press = (p[i+1][j] - p[i-1][j])*(0.5 / (DX * RHO));
                    const double v_press = (p[i][j+1] - p[i][j-1])*(0.5 / (DY * RHO));

                    /* time-step */
                    u_new[i][j] = u[i][j] + DT * (-u_conv + u_diff - u_press);
                    v_new[i][j] = v[i][j] + DT * (-v_conv + v_diff - v_press);
                }
            }
        }
    }

    /* ───── condições de contorno periódicas ──────────────────────── */
    #pragma omp parallel for
    for (int i = 0; i < NX; ++i) {
        u_new[i][0]     = u_new[i][NY-2];
        u_new[i][NY-1]  = u_new[i][1];
        v_new[i][0]     = v_new[i][NY-2];
        v_new[i][NY-1]  = v_new[i][1];
    }

    #pragma omp parallel for
    for (int j = 0; j < NY; ++j) {
        u_new[0][j]     = u_new[NX-2][j];
        u_new[NX-1][j]  = u_new[1][j];
        v_new[0][j]     = v_new[NX-2][j];
        v_new[NX-1][j]  = v_new[1][j];
    }

    /* ───── cópia de volta para os arrays originais ───────────────── */
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < NX; ++i)
        for (int j = 0; j < NY; ++j) {
            u[i][j] = u_new[i][j];
            v[i][j] = v_new[i][j];
        }
}

// Função para salvar os resultados em um arquivo (não paralelizado - I/O)
void save_results(int step, double p[NX][NY]) {
    char filename[100];
    sprintf(filename, "wave_%04d.dat", step);
    FILE *file = fopen(filename, "w");

    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            fprintf(file, "%d %d %f\n", i, j, p[i][j]);
        }
        fprintf(file, "\n");
    }

    fclose(file);
}

