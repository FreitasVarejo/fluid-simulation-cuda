// src/solver_cuda.cu

#include <cuda_runtime.h>
#include <math.h>
#include "solver_cuda.h"
#include "params.h"


__global__ void initialize_kernel(double *u, double *v, double *p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < NX && j < NY) {
        int idx = i * NY + j;
        u[idx] = 0.0;
        v[idx] = 0.0;
        double dist = sqrtf((i - NX/2)*(i - NX/2) + (j - NY/2)*(j - NY/2));
        p[idx] = (dist < 10.0) ? 10.0 * exp(-dist*dist/25.0) : 0.0;
    }
}


// Kernel para atualizar as velocidades
__global__ void update_velocities_kernel(double * __restrict__ u, double * __restrict__ v,
                              double * __restrict__ p,
                              double * __restrict__ u_new,
                              double * __restrict__ v_new)
{
    __shared__ double su[ (BLOCK_Y+2)*(BLOCK_X+2) ];
    __shared__ double sv[ (BLOCK_Y+2)*(BLOCK_X+2) ];
    __shared__ double sp[ (BLOCK_Y+2)*(BLOCK_X+2) ];

    const int gi = blockIdx.x*BLOCK_X + threadIdx.x;   // global indices
    const int gj = blockIdx.y*BLOCK_Y + threadIdx.y;
    const int li = threadIdx.x + 1;                    // local + halo offset
    const int lj = threadIdx.y + 1;
    const int lpitch = BLOCK_X+2;

    auto LINDEX = [&](int x,int y){ return (y)*lpitch + (x); };

    // ── load centre ──────────────────────────────────────────
    if (gi<NX && gj<NY){
        int gidx = gi*NY + gj;
        su[LINDEX(li,lj)] = u[gidx];
        sv[LINDEX(li,lj)] = v[gidx];
        sp[LINDEX(li,lj)] = p[gidx];
    }
    // ── load halo (4 sides, no corners) ─────────────────────
    // left / right halo
    if (threadIdx.x==0 && gi>0 && gj<NY){
        int gidxL=(gi-1)*NY+gj;
        su[LINDEX(li-1,lj)] = u[gidxL];
        sv[LINDEX(li-1,lj)] = v[gidxL];
        sp[LINDEX(li-1,lj)] = p[gidxL];
    }
    if (threadIdx.x==BLOCK_X-1 && gi+1<NX && gj<NY){
        int gidxR=(gi+1)*NY+gj;
        su[LINDEX(li+1,lj)] = u[gidxR];
        sv[LINDEX(li+1,lj)] = v[gidxR];
        sp[LINDEX(li+1,lj)] = p[gidxR];
    }
    // top / bottom halo
    if (threadIdx.y==0 && gj>0 && gi<NX){
        int gidxD=gi*NY + (gj-1);
        su[LINDEX(li,lj-1)] = u[gidxD];
        sv[LINDEX(li,lj-1)] = v[gidxD];
        sp[LINDEX(li,lj-1)] = p[gidxD];
    }
    if (threadIdx.y==BLOCK_Y-1 && gj+1<NY && gi<NX){
        int gidxU=gi*NY + (gj+1);
        su[LINDEX(li,lj+1)] = u[gidxU];
        sv[LINDEX(li,lj+1)] = v[gidxU];
        sp[LINDEX(li,lj+1)] = p[gidxU];
    }

    __syncthreads();

    if (gi>=1 && gi<NX-1 && gj>=1 && gj<NY-1){
        // local stencil reads only shared memory – fully in L1 / shared
        double u_c  = su[LINDEX(li,lj)];
        double v_c  = sv[LINDEX(li,lj)];
        double u_e  = su[LINDEX(li+1,lj)], u_w=su[LINDEX(li-1,lj)];
        double u_n  = su[LINDEX(li,lj+1)], u_s=su[LINDEX(li,lj-1)];
        double v_e  = sv[LINDEX(li+1,lj)], v_w=sv[LINDEX(li-1,lj)];
        double v_n  = sv[LINDEX(li,lj+1)], v_s=sv[LINDEX(li,lj-1)];
        double p_e  = sp[LINDEX(li+1,lj)], p_w=sp[LINDEX(li-1,lj)];
        double p_n  = sp[LINDEX(li,lj+1)], p_s=sp[LINDEX(li,lj-1)];

        // convection
        double u_conv = u_c*(u_e-u_w)*(0.5/ DX) + v_c*(u_n-u_s)*(0.5/ DY);
        double v_conv = u_c*(v_e-v_w)*(0.5/ DX) + v_c*(v_n-v_s)*(0.5/ DY);
        // diffusion
        double u_diff = NU*((u_e-2*u_c+u_w)/(DX*DX) + (u_n-2*u_c+u_s)/(DY*DY));
        double v_diff = NU*((v_e-2*v_c+v_w)/(DX*DX) + (v_n-2*v_c+v_s)/(DY*DY));
        // pressure
        double u_press = (p_e-p_w)*(0.5/(DX*RHO));
        double v_press = (p_n-p_s)*(0.5/(DY*RHO));

        int gidx = gi*NY + gj;
        u_new[gidx] = u_c + DT*(-u_conv + u_diff - u_press);
        v_new[gidx] = v_c + DT*(-v_conv + v_diff - v_press);
    }
    // periodic boundary copy pode ficar no host pós‑kernel, para simplificar.
}


// Função para resolver a equação de Poisson para pressão usando CUDA
__global__ void solve_pressure_kernel(const double *u, const double *v, const double *p, double *p_new)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i>=1 && i<NX-1 && j>=1 && j<NY-1) {
        int idx = i*NY + j;
        double div_u = (u[(i+1)*NY+j] - u[(i-1)*NY+j])/(2.0*DX)
                     + (v[i*NY+(j+1)] - v[i*NY+(j-1)])/(2.0*DY);

        p_new[idx] = ((p[(i+1)*NY+j] + p[(i-1)*NY+j])*DY*DY +
                     (p[i*NY+(j+1)] + p[i*NY+(j-1)])*DX*DX -
                     RHO*DX*DX*DY*DY*div_u/DT) /
                     (2.0*(DX*DX + DY*DY));
    }
}
