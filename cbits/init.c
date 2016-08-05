#include "cbits/stubs.h"
#include <stdio.h>

/*
 * Make sure that the linker always touches this module so that it notices the
 * below constructor function. Calling this empty function as part of
 * 'Foreign.CUDA.Driver.initialise' should be sufficient to prevent it from ever
 * being stripped.
 */
void enable_constructors() { }

/*
 * GHC-8 introduced a new (simpler) 64-bit allocator, which on startup 'mmap's
 * 1TB of address space and then commits sub-portions of that memory as needed.
 *
 * The CUDA driver also appears to 'mmap' a large chunk of address space on
 * 'cuInit', probably as the arena for shuffling memory to and from the device,
 * but attempts to do so at a _fixed_ address. If the GHC RTS has already taken
 * that address at the time we call 'cuInit', driver initialisation will fail
 * with an "out of memory" error.
 *
 * The workaround is to call 'cuInit' before initialising the RTS. Then the
 * RTS's allocation will avoid CUDA's allocation, since the RTS doesn't care
 * where in the address space it gets that memory. Embedding the following
 * __attribute__((constructor)) function in the library does the trick nicely,
 * and the linker will ensure that this gets executed when the shared library is
 * loaded (during program startup).
 *
 * Another way around this, without actually calling 'cuInit', would be to just
 * reserve the regions that 'cuInit' requires in the constructor function so
 * that the RTS avoids them, then release them before calling 'cuInit'. However,
 * since the CUDA driver is closed and we don't know exactly which regions to
 * reserve, that approach would be fragile.
 *
 * See: https://github.com/tmcdonell/cuda/issues/39
 */
__attribute__((constructor)) void preinitialise_cuda()
{
    CUresult status = cuInit (0);

    if ( status != CUDA_SUCCESS ) {
#if CUDA_VERSION >= 6000
        const char* str = NULL;

        cuGetErrorString(status, &str);
        fprintf(stderr, "Failed to pre-initialise CUDA: %s\n", str);
#else
        fprintf(stderr, "Failed to pre-initialise CUDA (%d)\n", status);
#endif
    }
}

