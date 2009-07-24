/*
 * Extra bits for C binding
 */

#ifndef C_STUBS_H
#define C_STUBS_H

#include <cuda_runtime_api.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Enums
 */
typedef struct cudaDeviceProp   cudaDeviceProp;

typedef enum
{
    cudaDeviceFlagScheduleAuto      = cudaDeviceScheduleAuto,
    cudaDeviceFlagScheduleSpin      = cudaDeviceScheduleSpin,
    cudaDeviceFlagScheduleYield     = cudaDeviceScheduleYield,
    cudaDeviceFlagBlockingSync      = cudaDeviceBlockingSync,
    cudaDeviceFlagMapHost           = cudaDeviceMapHost
} cudaDeviceFlags;


/*
 * Functions
 */


#ifdef __cplusplus
}
#endif
#endif
