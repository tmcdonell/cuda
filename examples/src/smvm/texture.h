/*
 *  Copyright 2008-2009 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */


#ifndef __TEXTURE_H__
#define __TEXTURE_H__

#include "utils.h"
#include <cuda_runtime_api.h>

/*
 * These textures are (optionally) used to cache the 'x' vector in y += A*x
 * Use int2 to pull doubles through texture cache.
 */
texture<float,1> tex_x_float;
texture<int2,1>  tex_x_double;

inline void
bind_x(const float * x)
{
    size_t offset = size_t(-1);

    CUDA_SAFE_CALL(cudaBindTexture(&offset, tex_x_float, x));
    if (offset != 0)
        assert(!"memory is not aligned, refusing to use texture cache");
}

inline void
bind_x(const double * x)
{
    size_t offset = size_t(-1);

    CUDA_SAFE_CALL(cudaBindTexture(&offset, tex_x_double, x));
    if (offset != 0)
        assert(!"memory is not aligned, refusing to use texture cache");
}

/*
 * NOTE: the parameter is unused only to distinguish the two unbind functions
 */
inline void
unbind_x(const float *)
{
    CUDA_SAFE_CALL(cudaUnbindTexture(tex_x_float));
}

inline void
unbind_x(const double *)
{
    CUDA_SAFE_CALL(cudaUnbindTexture(tex_x_double));
}

template <bool UseCache>
__inline__ __device__ float
fetch_x(const int& i, const float * x)
{
    if (UseCache) return tex1Dfetch(tex_x_float, i);
    else          return x[i];
}

#ifndef CUDA_NO_SM_13_DOUBLE_INTRINSICS
template <bool UseCache>
__inline__ __device__ double fetch_x(const int& i, const double * x)
{
#if __CUDA_ARCH__ < 130
#error "double precision require Compute Compatibility 1.3 or greater"
#endif
    if (UseCache)
    {
        int2 v = tex1Dfetch(tex_x_double, i);
        return __hiloint2double(v.y, v.x);
    }
    else
    {
        return x[i];
    }
}
#endif

#endif  // __TEXTURE_H__

