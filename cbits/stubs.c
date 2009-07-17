/*
 * Extra bits for C binding
 */

#include "stubs.h"

/*
 * DeviceProperties
 *
 * Required because the C->Haskell get hook deferences static arrays but still
 * treats them as pointer type.
 */
int offset_devName()
{
    return offsetof(struct cudaDeviceProp, name);
}

int offset_devMaxThreadDim()
{
    return offsetof(struct cudaDeviceProp, maxThreadsDim);
}

int offset_devMaxGridSize()
{
    return offsetof(struct cudaDeviceProp, maxGridSize);
}

