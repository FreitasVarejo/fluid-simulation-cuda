#pragma once
#ifdef __CUDACC__
/* protótipo visível para host e device - evita mangling divergente */
__global__ void update_velocities_kernel(double *__restrict__ u,
                                         double *__restrict__ v,
                                         double *__restrict__ p,
                                         double *__restrict__ u_new,
                                         double *__restrict__ v_new);
#endif
