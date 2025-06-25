#ifndef SOLVER_H
#define SOLVER_H

#include "params.h"

void initialize(double u[NX][NY], double v[NX][NY], double p[NX][NY]);
void solve_pressure(double u[NX][NY], double v[NX][NY], double p[NX][NY]);
void update_velocities(double u[NX][NY], double v[NX][NY], double p[NX][NY]);
void save_results(int step, double p[NX][NY]);

#endif
