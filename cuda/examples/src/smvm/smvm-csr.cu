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


#include "smvm.h"
#include "utils.h"
#include "texture.h"


/* -----------------------------------------------------------------------------
 * Sparse-matrix dense-vector multiplication, compressed-sparse row format
 * -----------------------------------------------------------------------------
 *
 * Each row of the CSR matrix is assigned to a warp, which computes the dot
 * product of the i-th row of A with the x vector, in parallel.
 *
 *   y[i] = A[i,:] * x
 *
 * This division of work implies that the CSR index and data arrays (Aj and Ax)
 * are accessed in a contiguous manner (but generally not aligned). On the GT200
 * these accesses are coalesced, unlike kernels based on the one-row-per-thread
 * division of work. Since an entire 32-thread warp is assigned to each row,
 * many threads will remain idle when their row contains a small number of
 * elements. This code relies on implicit synchronization among threads in a
 * warp.
 *
 * Optionally, the texture cache may be used for accessing the x vector. This
 * generally shows good improvements.
 *
 * References:
 *
 *  [1] N. Bell and M. Garland. "Efficient sparse matrix-vector multiplication on
 *      CUDA." NVIDIA Technical Report NVR-2008-004, NVIDIA Corporation, Dec. 2008.
 *
 *  [2] N. Bell and M. Garland. "Implementing sparse matrix-vector
 *      multiplication on throughput-oriented processors." In Supercomputing
 *      `09: Proceedings of the 2009 Conference on High Performance Computing
 *      Networking, Storage and Analysis, pages 1-11, 2009.
 */
template <unsigned int BlockSize, typename T, bool UseCache>
__global__ static void
smvm_k
(
    T                   *d_y,
    const T             *d_x,
    const T             *d_Ax,
    const unsigned int  *d_Ap,
    const unsigned int  *d_Aj,
    const unsigned int  num_rows
)
{
    /*
     * Require at least a full warp for each row. This could be relaxed by
     * modifying the cooperative reduction step
     */
    assert(BlockSize % WARP_SIZE == 0);

    const unsigned int vectorsPerBlock = BlockSize / WARP_SIZE;
    const unsigned int num_vectors     = vectorsPerBlock * gridDim.x;
    const unsigned int thread_id       = BlockSize * blockIdx.x + threadIdx.x;
    const unsigned int vector_id       = thread_id / WARP_SIZE;
    const unsigned int thread_lane     = threadIdx.x & (WARP_SIZE-1);
    const unsigned int vector_lane     = threadIdx.x / WARP_SIZE;

    __shared__ volatile T            s_data[(vectorsPerBlock+1) * WARP_SIZE];
    __shared__ volatile unsigned int s_ptrs[vectorsPerBlock][2];

    for (unsigned int row = vector_id; row < num_rows; row += num_vectors)
    {
        /*
         * Use two threads to fetch the indices of the start and end of this
         * segment. This is a single coalesced (although unaligned) global read
         * rather than two, and hence considerably faster.
         */
        if (thread_lane < 2)
            s_ptrs[vector_lane][thread_lane] = d_Ap[row + thread_lane];

        __EMUSYNC;
        const unsigned int row_start = s_ptrs[vector_lane][0];
        const unsigned int row_end   = s_ptrs[vector_lane][1];

        /*
         * Have the threads read in all values for this row, accumulating local
         * dot-product sums. Then, reduce this cooperatively in shared memory.
         */
        T sum = 0;
        for (unsigned int j = row_start + thread_lane; j < row_end; j += WARP_SIZE)
            sum += d_Ax[j] * fetch_x<UseCache>(d_Aj[j], d_x);

        s_data[threadIdx.x] = sum;                                  __EMUSYNC;
        s_data[threadIdx.x] = sum = sum + s_data[threadIdx.x + 16]; __EMUSYNC;
        s_data[threadIdx.x] = sum = sum + s_data[threadIdx.x +  8]; __EMUSYNC;
        s_data[threadIdx.x] = sum = sum + s_data[threadIdx.x +  4]; __EMUSYNC;
        s_data[threadIdx.x] = sum = sum + s_data[threadIdx.x +  2]; __EMUSYNC;
        s_data[threadIdx.x] = sum = sum + s_data[threadIdx.x +  1]; __EMUSYNC;

#if 0
        /*
         * Alternative method (slightly slower, due to bank conflicts?)
         */
        s_data[threadIdx.x] += s_data[threadIdx.x + 16];
        s_data[threadIdx.x] += s_data[threadIdx.x +  8];
        s_data[threadIdx.x] += s_data[threadIdx.x +  4];
        s_data[threadIdx.x] += s_data[threadIdx.x +  2];
        s_data[threadIdx.x] += s_data[threadIdx.x +  1];
#endif

        /*
         * Finally, first thread writes the result for this row
         */
        if (thread_lane == 0)
            d_y[row] = s_data[threadIdx.x];
    }
}


template <typename T, bool UseCache>
static void
smvm_dispatch
(
    T                   *d_y,
    const T             *d_x,
    const T             *d_data,
    const unsigned int  *d_ptr,
    const unsigned int  *d_indices,
    const unsigned int  num_rows,
    const unsigned int  blocks,
    const unsigned int  threads
)
{
    const unsigned int smem = 0;

    switch (threads)
    {
    case 512: smvm_k<512,T,UseCache><<<blocks,threads,smem>>>(d_y, d_x, d_data, d_ptr, d_indices, num_rows); break;
    case 256: smvm_k<256,T,UseCache><<<blocks,threads,smem>>>(d_y, d_x, d_data, d_ptr, d_indices, num_rows); break;
    case 128: smvm_k<128,T,UseCache><<<blocks,threads,smem>>>(d_y, d_x, d_data, d_ptr, d_indices, num_rows); break;
    case  64: smvm_k< 64,T,UseCache><<<blocks,threads,smem>>>(d_y, d_x, d_data, d_ptr, d_indices, num_rows); break;
    case  32: smvm_k< 32,T,UseCache><<<blocks,threads,smem>>>(d_y, d_x, d_data, d_ptr, d_indices, num_rows); break;
    default:
        assert(!"Non-exhaustive patterns in match");
    }
}


/*
 * Select an "optimal" number of threads and blocks for the problem size. This
 * is an act of balancing resource usage: shared memory, registers, in-flight
 * threads and blocks per multiprocessor. Ultimately, this requires some
 * experimentation for every kernel, device and problem set, but we choose some
 * sensible default values.
 *
 * Additionally, each block will have at least one full warp, as required by the
 * core kernel.
 */
static void
smvm_control
(
    unsigned int        n,
    unsigned int        &blocks,
    unsigned int        &threads,
    unsigned int        maxThreads = MAX_THREADS,
    unsigned int        maxBlocks  = MAX_BLOCKS

)
{
    threads = (n < maxThreads) ? max(WARP_SIZE, ceilPow2(n)) : maxThreads;
    blocks  = (n + threads - 1) / threads;
    blocks  = min(blocks, maxBlocks);
}


/*
 * Sparse matrix multiplication:
 *   y = A * x
 *
 * The CSR format explicitly stores column indices (indices) and non-zero values
 * (data) in row-major order, together with a third array of row pointers (ptr).
 * For an M-by-N matrix, ptr has length (M+1) and stores the offset to the start
 * of the i-th row in ptr[i]. The last entry then, corresponding to the (M+1)-st
 * row, contains the number of non-zero elements in the matrix.
 *
 * Example:
 *            | 1 7 0 0 |
 *        A = | 0 2 8 0 |
 *            | 5 0 3 9 |
 *            | 0 6 0 4 |
 *
 *      ptr = [ 0 2 4 7 9 ]
 *  indices = [ 0 1 1 2 0 2 3 1 3 ]
 *     data = [ 1 7 2 8 5 3 9 6 4 ]
 *
 * d_y          The output vector
 * d_x          The input (dense) vector to multiply against
 * d_data       The non-zero elements of the sparse, stored row-major order
 * d_ptr        Row offsets
 * d_indices    Column indices
 */
template <typename T, bool UseCache>
static void
smvm_csr
(
    T                   *d_y,
    const T             *d_x,
    const T             *d_data,
    const unsigned int  *d_ptr,
    const unsigned int  *d_indices,
    const unsigned int  num_rows
)
{
    unsigned int blocks;
    unsigned int threads;

    if (UseCache)
        bind_x(d_x);

    smvm_control(num_rows, blocks, threads);
    smvm_dispatch<T,UseCache>(d_y, d_x, d_data, d_ptr, d_indices, num_rows, blocks, threads);

    if (UseCache)
        unbind_x(d_x);
}

/* -----------------------------------------------------------------------------
 * Instances
 * ---------------------------------------------------------------------------*/

void
smvm_csr_f(float *d_y, float *d_x, float *d_data, unsigned int *d_rowPtr, unsigned int *d_colIdx, unsigned int num_rows)
{
    smvm_csr<float,true>(d_y, d_x, d_data, d_rowPtr, d_colIdx, num_rows);
}

