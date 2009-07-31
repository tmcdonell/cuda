#ifndef __KERNELS_H
#define __KERNELS_H

extern __global__ void plus_scan_kernel(float *g_out, float *g_in, int n);

#endif
