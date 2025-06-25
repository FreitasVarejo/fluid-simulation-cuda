// src/main_cuda.cu

#include <stdio.h>
#include "solver_cuda.h"
#include <cuda_runtime.h>
#include "params.h"
#include <stdlib.h>
#include "helpers.h"

#ifndef BLOCK_X
#define BLOCK_X 32        // defaults p/ build sem -DBLOCK_X
#endif
#ifndef BLOCK_Y
#define BLOCK_Y 32
#endif

#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(msg)                                             \
    do {                                                            \
        cudaError_t e = cudaGetLastError();                         \
        if (e != cudaSuccess) {                                     \
            fprintf(stderr, "CUDA-ERROR (%s): %s\n", (msg),         \
                    cudaGetErrorString(e));                         \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

int main() {
    double *d_u, *d_v, *d_p, *d_p_new, *d_u_new, *d_v_new;
    size_t size = NX * NY * sizeof(double);

    // Pega envs
    const char *variant = getenv_or("WAVE_VARIANT", "cuda");
    const char *cfg     = getenv_or("WAVE_CFG",     "N100_T32_TB1");
    const char *param   = getenv_or("WAVE_PARAM",   "-");

    // Host buffer para salvar p
    double *p_host = (double*)malloc(size);

    cudaMalloc(&d_u, size);  cudaMalloc(&d_v, size);
    cudaMalloc(&d_p, size);  cudaMalloc(&d_p_new, size);
    cudaMalloc(&d_u_new, size);  cudaMalloc(&d_v_new, size);

    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid((NX + BLOCK_X - 1) / BLOCK_X,
              (NY + BLOCK_Y - 1) / BLOCK_Y);

    initialize_kernel<<<grid, block>>>(d_u, d_v, d_p);
    CUDA_CHECK("init");
    cudaDeviceSynchronize();

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);  cudaEventCreate(&t1);
    cudaEventRecord(t0);

    for (int t = 0; t < NT; ++t) {
        solve_pressure_kernel<<<grid, block>>>(d_u, d_v, d_p, d_p_new);
        CUDA_CHECK("pressure");

        update_velocities_kernel<<<grid, block>>>(d_u, d_v, d_p,
                                                  d_u_new, d_v_new);
        CUDA_CHECK("vel");

        cudaMemcpy(d_u, d_u_new, size, cudaMemcpyDeviceToDevice);
        CUDA_CHECK("copy u");
        cudaMemcpy(d_v, d_v_new, size, cudaMemcpyDeviceToDevice);
        CUDA_CHECK("copy v");
        cudaMemcpy(d_p, d_p_new, size, cudaMemcpyDeviceToDevice);
        CUDA_CHECK("copy p");

        if (t % 10 == 0) {
            cudaMemcpy(p_host, d_p, size, cudaMemcpyDeviceToHost);
            save_results_cuda(variant, cfg, param, t / 10, p_host);
        }
    }

    cudaEventRecord(t1);  cudaEventSynchronize(t1);
    float ms;
    cudaEventElapsedTime(&ms, t0, t1);
    printf("cuda:   %.9f\n", ms / 1e3);

    // cleanup
    cudaFree(d_u); cudaFree(d_v); cudaFree(d_p); cudaFree(d_p_new);
    cudaFree(d_u_new); cudaFree(d_v_new);
    free(p_host);
    return 0;
}
