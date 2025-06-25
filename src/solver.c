// src/solver.c


#include <sys/stat.h>
#include <stdlib.h>
#include <string.h>
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
/* ------------------------------------------------------------------------- */
void solve_pressure(double u[NX][NY],
                    double v[NX][NY],
                    double p[NX][NY])
{
    static double p_new[NX][NY];
    const int    max_iter = 1000;
    const double tol      = 1e-6;

    /* cópia inicial ------------------------------------------------- */
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < NX; ++i)
        for (int j = 0; j < NY; ++j)
            p_new[i][j] = p[i][j];

    for (int it = 0; it < max_iter; ++it) {

        double max_err = 0.0;                 /* agora é SHARED        */

        #pragma omp parallel for collapse(2) reduction(max:max_err)
        for (int i = 1; i < NX - 1; ++i)
            for (int j = 1; j < NY - 1; ++j) {

                const double div_u =
                      (u[i+1][j] - u[i-1][j]) * (0.5 / DX)
                    + (v[i][j+1] - v[i][j-1]) * (0.5 / DY);

                const double p_old = p_new[i][j];

                p_new[i][j] = ( (p_new[i+1][j] + p_new[i-1][j]) * DY*DY +
                                (p_new[i][j+1] + p_new[i][j-1]) * DX*DX -
                                RHO * DX*DX * DY*DY * div_u / DT ) /
                               (2.0 * (DX*DX + DY*DY));

                double err = fabs(p_new[i][j] - p_old);
                if (err > max_err) max_err = err;   /* redução faz o resto */
            }

        if (max_err < tol) break;             /* convergiu? ---------- */

        /* fronteiras de Neumann (gradiente zero) -------------------- */
        #pragma omp parallel for
        for (int i = 0; i < NX; ++i) {
            p_new[i][0]    = p_new[i][1];
            p_new[i][NY-1] = p_new[i][NY-2];
        }
        #pragma omp parallel for
        for (int j = 0; j < NY; ++j) {
            p_new[0][j]    = p_new[1][j];
            p_new[NX-1][j] = p_new[NX-2][j];
        }
    }

    /* cópia de volta ------------------------------------------------ */
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < NX; ++i)
        for (int j = 0; j < NY; ++j)
            p[i][j] = p_new[i][j];
}

void update_velocities(double u[NX][NY],
                       double v[NX][NY],
                       double p[NX][NY])
{
    static double u_new[NX][NY] __attribute__((aligned(64)));
    static double v_new[NX][NY] __attribute__((aligned(64)));

    #pragma omp parallel
    {
        /* varredura em blocos (tiling) – paralelo só no nível externo ------ */
        for (int ii = 1; ii < NX - 1; ii += TILE)
            for (int jj = 1; jj < NY - 1; jj += TILE) {

                const int i_max = (ii + TILE < NX - 1) ? ii + TILE : NX - 1;
                const int j_max = (jj + TILE < NY - 1) ? jj + TILE : NY - 1;

                #pragma omp for schedule(static)
                for (int i = ii; i < i_max; ++i) {

                    #pragma omp simd
                    for (int j = jj; j < j_max; ++j) {

                        /* convecção */
                        const double u_conv = u[i][j]*(u[i+1][j]-u[i-1][j])*(0.5/DX)
                                            + v[i][j]*(u[i][j+1]-u[i][j-1])*(0.5/DY);

                        const double v_conv = u[i][j]*(v[i+1][j]-v[i-1][j])*(0.5/DX)
                                            + v[i][j]*(v[i][j+1]-v[i][j-1])*(0.5/DY);

                        /* difusão */
                        const double u_diff = NU*((u[i+1][j]-2*u[i][j]+u[i-1][j])/(DX*DX) +
                                                  (u[i][j+1]-2*u[i][j]+u[i][j-1])/(DY*DY));

                        const double v_diff = NU*((v[i+1][j]-2*v[i][j]+v[i-1][j])/(DX*DX) +
                                                  (v[i][j+1]-2*v[i][j]+v[i][j-1])/(DY*DY));

                        /* pressão */
                        const double u_press = (p[i+1][j]-p[i-1][j])*(0.5/(DX*RHO));
                        const double v_press = (p[i][j+1]-p[i][j-1])*(0.5/(DY*RHO));

                        /* passo de tempo */
                        u_new[i][j] = u[i][j] + DT*(-u_conv + u_diff - u_press);
                        v_new[i][j] = v[i][j] + DT*(-v_conv + v_diff - v_press);
                    }
                }
            }

        /* condições de contorno periódicas --------------------------------- */
        #pragma omp for
        for (int i = 0; i < NX; ++i) {
            u_new[i][0]     = u_new[i][NY-2];
            u_new[i][NY-1]  = u_new[i][1];
            v_new[i][0]     = v_new[i][NY-2];
            v_new[i][NY-1]  = v_new[i][1];
        }
        #pragma omp for
        for (int j = 0; j < NY; ++j) {
            u_new[0][j]     = u_new[NX-2][j];
            u_new[NX-1][j]  = u_new[1][j];
            v_new[0][j]     = v_new[NX-2][j];
            v_new[NX-1][j]  = v_new[1][j];
        }

        /* cópia de volta --------------------------------------------------- */
        #pragma omp for collapse(2)
        for (int i = 0; i < NX; ++i)
            for (int j = 0; j < NY; ++j) {
                u[i][j] = u_new[i][j];
                v[i][j] = v_new[i][j];
            }
    }
}

void save_results(const char *variant, const char *cfg, const char *param, int step, double p[NX][NY]) {
    char folder[256];
    sprintf(folder, "data/%s/%s/%s", variant, cfg, param);

    // cria diretórios se não existirem
    char cmd[300];
    sprintf(cmd, "mkdir -p %s", folder);
    system(cmd);

    // monta caminho do arquivo
    char filename[300];
    sprintf(filename, "%s/wave_%04d.dat", folder, step);
    FILE *file = fopen(filename, "w");

    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            fprintf(file, "%d %d %f\n", i, j, p[i][j]);
        }
        fprintf(file, "\n");
    }
    fclose(file);
}
