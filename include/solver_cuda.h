// solver_cuda.h
#pragma once

#ifdef __CUDACC__
__global__ void initialize_kernel(double*, double*, double*);
__global__ void update_velocities_kernel(double*, double*, double*, double*, double*);
__global__ void solve_pressure_kernel(const double*, const double*, const double*, double*);
#endif

#ifdef __cplusplus
extern "C" {
#endif

void save_results_cuda(const char *variant,
                       const char *cfg,
                       const char *param,
                       int   step,
                       const double *p_host);

#ifdef __cplusplus
}
#endif
