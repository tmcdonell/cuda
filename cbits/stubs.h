/*
 * Extra bits for C binding
 */

#ifndef C_STUBS_H
#define C_STUBS_H

#include <cuda_runtime_api.h>

typedef struct cudaDeviceProp   cudaDeviceProp;

typedef enum
{
    cudaDeviceFlagScheduleAuto      = cudaDeviceScheduleAuto,
    cudaDeviceFlagScheduleSpin      = cudaDeviceScheduleSpin,
    cudaDeviceFlagScheduleYield     = cudaDeviceScheduleYield,
    cudaDeviceFlagBlockingSync      = cudaDeviceBlockingSync,
    cudaDeviceFlagMapHost           = cudaDeviceMapHost
} cudaDeviceFlags;

#endif
