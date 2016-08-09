#define _GNU_SOURCE

#include <cuda.h>

#include <stdint.h>
#include <stdio.h>
#include <unistd.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>


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
 * One workaround is to call 'cuInit' before initialising the RTS. Then the
 * RTS's allocation will avoid CUDA's allocation, since the RTS doesn't care
 * where in the address space it gets that memory. Embedding the following
 * __attribute__((constructor)) function in the library does the trick nicely,
 * and the linker will ensure that this gets executed when the shared library is
 * loaded (during program startup).
 *
 * However, this forces us to initialise the CUDA driver even if we would never
 * have used it, and precludes setting custom initialisation options (even
 * though the option flags to 'cuInit' are currently ignored). Instead, we
 * simply reserve the memory region that 'cuInit' requires in the constructor
 * function so that the RTS avoids it, and override the 'cuInit' symbol with our
 * own version that releases that region before calling the original 'cuInit'.
 * The disadvantage of course is that this is a little fragile, since the CUDA
 * driver is closed and we don't know exactly which regions to reserve.
 *
 * See: https://github.com/tmcdonell/cuda/issues/39
 */
static int*      __reserved               = NULL;
static uintptr_t __reserved_regions[1][2] = {{0x200000000, 0x1000000000}};

/*
 * Reserve the CUDA memory range so that the GHC RTS avoids it.
 *
 * The magic numbers in '__reserved_regions' found by running under gdb and
 * calling 'info proc mappings' before and after the call to 'cuInit'. These may
 * not be the only regions that need to be reserved...
 */
void reserve_cuda_memory_region()
{
    msync(__reserved, getpagesize(), MS_SYNC);

    if ( !(*__reserved) ) {
        int i;
        for (i = 0; i < 1; ++i) {
            void *result = mmap( (void*) __reserved_regions[i][0], __reserved_regions[i][1]
                               , PROT_NONE
                               , MAP_PRIVATE | MAP_FIXED | MAP_NORESERVE | MAP_ANONYMOUS
                               , -1
                               , 0
                               );

            if ( MAP_FAILED == result ) {
                perror("Failed to reserve CUDA memory region");
                return;
            }
        }
        *__reserved = 1;
    }
}

/*
 * Once the GHC RTS has been initialised, we can release the memory region and
 * 'cuInit' should proceed as usual
 */
void release_cuda_memory_region()
{
    msync(__reserved, getpagesize(), MS_SYNC);

    if ( *__reserved ) {
        int i;
        for (i = 0; i < 1; ++i) {
            int result = munmap( (void*) __reserved_regions[i][0], __reserved_regions[i][1] );

            if ( 0 != result ) {
                perror("Failed to release reserved CUDA memory region");
                return;
            }
        }
        *__reserved = 0;
    }
}

/*
 * Transparently replace NVIDIA's implementation of 'cuInit' with our own, which
 * releases the reserved memory region, if necessary, before calling the real
 * 'cuInit'.
 *
 * Sadly this doesn't appear to override the call to 'cuInit' from deep within
 * the (statically linked) runtime API, so users of the runtime API will need to
 * explicitly 'initialise' their programs now.
 */
CUresult CUDAAPI cuInit(unsigned int Flags)
{
    release_cuda_memory_region();

    CUresult CUDAAPI (*original_cuInit)(unsigned int) = dlsym(RTLD_NEXT, "cuInit");
    return original_cuInit(Flags);
}


#define SHM_FILE "/hscuda.reserved.smkey"

/*
* Every process linked against a library will get its own copy of the
* global/static variables defined by that library. In this case this is not what
* we want, since we want to signal to _all_ processes whether or not the CUDA
* regions have been reserved or not, so instead we create a shared variable to
* signal this. Then, only the first loaded instance of the library reserves the
* memory block, and only when [our version of] 'cuInit' is called is that memory
* released for use by the CUDA driver.
*
* We don't expect this to be run concurrently, so a mutex is unnecessary.
*/
__attribute__((constructor)) void __hscuda_setup()
{
    /*
     * Create or open the shared memory region. If it does not exist we create
     * it and set the size of the newly allocated object, which also initialises
     * it to zero. If it does exist, open the existing region.
     */
    int fd = shm_open(SHM_FILE, O_RDWR | O_CREAT | O_EXCL, S_IRUSR | S_IWUSR);
    if ( fd > 0 ) {
        /*
         * Set the size of the newly created object. Must be a multiple of the
         * page size.
         */
        int err = ftruncate(fd, getpagesize());
        if ( err < 0 ) {
            perror("Failed to allocate shared memory object");
            exit(EXIT_FAILURE);
        }
    } else {
        /*
         * Attempt to open to the existing shared-memory region
         */
        fd = shm_open(SHM_FILE, O_RDWR, S_IRUSR | S_IWUSR);
        if ( fd < 0 ) {
            perror("Failed to open existing shared memory object");
            exit(EXIT_FAILURE);
        }
    }

    /*
     * Map the shared memory segment into the address space of the process
     */
    __reserved = mmap(NULL, getpagesize(), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if ( MAP_FAILED == __reserved ) {
        perror("Failed to reserve shared memory");
        exit(EXIT_FAILURE);
    }

    reserve_cuda_memory_region();
}

__attribute__((destructor)) void __hscuda_teardown()
{
    release_cuda_memory_region();

    munmap(__reserved, getpagesize());
    shm_unlink(SHM_FILE);
}

#undef SHM_FILE

