/*
 * Extra bits for C binding
 */

#ifndef C_STUBS_H
#define C_STUBS_H

#include <cuda_runtime_api.h>

typedef struct cudaDeviceProp cudaDeviceProp;

extern int offset_devName();
extern int offset_devMaxThreadDim();
extern int offset_devMaxGridSize();


#endif
