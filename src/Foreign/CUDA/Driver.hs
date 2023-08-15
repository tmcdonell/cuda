--------------------------------------------------------------------------------
-- |
-- Module    : Foreign.CUDA.Driver
-- Copyright : [2009..2023] Trevor L. McDonell
-- License   : BSD
--
-- This module defines an interface to the CUDA driver API. The Driver API
-- is a lower-level interface to CUDA devices than that provided by the
-- Runtime API. Using the Driver API, the programmer must deal explicitly
-- with operations such as initialisation, context management, and loading
-- (kernel) modules. Although more difficult to use initially, the Driver
-- API provides more control over how CUDA is used. Furthermore, since it
-- does not require compiling and linking the program with 'nvcc', the
-- Driver API provides better inter-language compatibility.
--
-- The following is a short tutorial on using the Driver API. The steps can
-- be copied into a file, or run directly in @ghci@, in which case @ghci@
-- should be launched with the option @-fno-ghci-sandbox@. This is because
-- CUDA maintains CPU-local state, so operations should always be run from
-- a bound thread.
--
--
-- [/Using the Driver API/]
--
-- Before any operation can be performed, the Driver API must be
-- initialised:
--
-- >>> import Foreign.CUDA.Driver
-- >>> initialise []
--
-- Next, we must select a GPU that we will execute operations on. Each GPU
-- is assigned a unique identifier (beginning at zero). We can get a handle
-- to a compute device at a given ordinal using the 'device' operation.
-- Given a device handle, we can query the properties of that device using
-- 'props'. The number of available CUDA-capable devices is given via
-- 'count'. For example:
--
-- >>> count
-- 1
-- >>> dev0 <- device 0
-- >>> props dev0
-- DeviceProperties {deviceName = "GeForce GT 650M", computeCapability = 3.0, ...}
--
-- This package also includes the executable 'nvidia-device-query', which when
-- executed displays the key properties of all available devices. See
-- "Foreign.CUDA.Driver.Device" for additional operations to query the
-- capabilities or status of a device.
--
-- Once you have chosen a device to use, the next step is to create a CUDA
-- context. A context is associated with a particular device, and all
-- operations, such as memory allocation and kernel execution, take place
-- with respect to that context. For example, to 'create' a new execution
-- context on CUDA device 0:
--
-- >>> ctx <- create dev0 []
--
-- The second argument is a set of 'ContextFlag's which control how the
-- context behaves in various situations, for example, whether or not the
-- CPU should actively spin when waiting for results from the GPU
-- ('SchedSpin'), or to yield control to other threads instead
-- ('SchedYield').
--
-- The newly created context is now the /active/ context, and all
-- subsequent operations take place within that context. More than one
-- context can be created per device, but resources, such as memory
-- allocated in the GPU, are unique to each context. The module
-- "Foreign.CUDA.Driver.Context" contains operations for managing multiple
-- contexts. Some devices allow data to be shared between contexts without
-- copying, see "Foreign.CUDA.Driver.Context.Peer" for more information.
--
-- Once the context is no longer needed, it should be 'destroy'ed in order
-- to free up any resources that were allocated to it.
--
-- >>> destroy ctx
--
-- Each device also has a unique context which is used by the Runtime API.
-- This context can be accessed with the module
-- "Foreign.CUDA.Driver.Context.Primary".
--
--
-- [/Executing kernels onto the GPU/]
--
-- Once the Driver API is initialised and an execution context is created
-- on the GPU, we can begin to interact with it.
--
-- At an example, we'll step through executing the CUDA equivalent of the
-- following Haskell function, which element-wise adds the elements of two
-- arrays:
--
-- >>> vecAdd xs ys = zipWith (+) xs ys
--
-- The following CUDA kernel can be used to implement this on the GPU:
--
-- > extern "C" __global__ void vecAdd(float *xs, float *ys, float *zs, int N)
-- > {
-- >     int ix = blockIdx.x * blockDim.x + threadIdx.x;
-- >
-- >     if ( ix < N ) {
-- >         zs[ix] = xs[ix] + ys[ix];
-- >     }
-- > }
--
-- Here, the @__global__@ keyword marks the function as a kernel that
-- should be computed on the GPU in data parallel. When we execute this
-- function on the GPU, (at least) /N/ threads will execute /N/ individual
-- instances of the kernel function @vecAdd@. Each thread will operate on
-- a single element of each input array to create a single value in the
-- result. See the CUDA programming guide for more details.
--
-- We can save this to a file @vector_add.cu@, and compile it using @nvcc@
-- into a form that we can then load onto the GPU and execute:
--
-- > $ nvcc --ptx vector_add.cu
--
-- The module "Foreign.CUDA.Driver.Module" contains functions for loading
-- the resulting .ptx file (or .cubin files) into the running program.
--
-- >>> mdl <- loadFile "vector_add.ptx"
--
-- Once finished with the module, it is also a good idea to 'unload' it.
--
-- Modules may export kernel functions, global variables, and texture
-- references. Before we can execute our function, we need to look it up in
-- the module by name.
--
-- >>> vecAdd <- getFun mdl "vecAdd"
--
-- Given this reference to our kernel function, we are almost ready to
-- execute it on the device using 'launchKernel', but first, we must create
-- some data that we can execute the function on.
--
--
-- [/Transferring data to and from the GPU/]
--
-- GPUs typically have their own memory which is separate from the CPU's
-- memory, and we need to explicitly copy data back and forth between these
-- two regions. The module "Foreign.CUDA.Driver.Marshal" provides functions
-- for allocating memory on the GPU, and copying data between the CPU and
-- GPU, as well as directly between multiple GPUs.
--
-- For simplicity, we'll use standard Haskell lists for our input and
-- output data structure. Note however that this will have significantly
-- lower effective bandwidth than reading a single contiguous region of
-- memory, so for most practical purposes you will want to use some kind of
-- unboxed array.
--
-- >>> let xs = [1..1024]   :: [Float]
-- >>> let ys = [2,4..2048] :: [Float]
--
-- In CUDA, like C, all memory management is explicit, and arrays on the
-- device must be explicitly allocated and freed. As mentioned previously,
-- data transfer is also explicit. However, we do provide convenience
-- functions for combined allocation and marshalling, as well as bracketed
-- operations.
--
-- >>> xs_dev <- newListArray xs
-- >>> ys_dev <- newListArray ys
-- >>> zs_dev <- mallocArray 1024 :: IO (DevicePtr Float)
--
-- After executing the kernel (see next section), we transfer the result
-- back to the host, and free the memory that was allocated on the GPU.
--
-- >>> zs <- peekListArray 1024 zs_dev
-- >>> free xs_dev
-- >>> free ys_dev
-- >>> free zs_dev
--
--
-- [/Piecing it all together/]
--
-- Finally, we have everything in place to execute our operation on the
-- GPU. Launching a kernel on the GPU consists of creating many threads on
-- the GPU which all execute the same function, and each thread has
-- a unique identifier in the grid/block hierarchy which can be used to
-- identify exactly which element this thread should process (the
-- @blockIdx@ and @threadIdx@ parameters that we saw earlier,
-- respectively).
--
-- To execute our function, we will use a grid of 4 blocks, each containing
-- 256 threads. Thus, a total of 1024 threads will be launched, which will
-- each compute a single element of the output array (recall that our input
-- arrays each have 1024 elements). The module
-- "Foreign.CUDA.Analysis.Occupancy" contains functions to help determine
-- the ideal thread block size for a given kernel and GPU combination.
--
-- >>> launchKernel vecAdd (4,1,1) (256,1,1) 0 Nothing [VArg xs_dev, VArg ys_dev, VArg zs_dev, IArg 1024]
--
-- Note that kernel execution is asynchronous, so we should also wait for
-- the operation to complete before attempting to read the results back.
--
-- >>> sync
--
-- And that's it!
--
--
-- [/Next steps/]
--
-- As mentioned at the end of the previous section, kernels on the GPU are
-- executed asynchronously with respect to the host, and other operations
-- such as data transfers can also be executed asynchronously. This allows
-- the CPU to continue doing other work while the GPU is busy.
-- 'Foreign.CUDA.Driver.Event.Event's can be used to check whether an
-- operation has completed yet.
--
-- It is also possible to execute multiple kernels or data transfers
-- concurrently with each other, by assigning those operations to different
-- execution 'Foreign.CUDA.Driver.Stream.Stream's. Used in conjunction with
-- 'Foreign.CUDA.Driver.Event.Event's, operations will be scheduled
-- efficiently only once all dependencies (in the form of
-- 'Foreign.CUDA.Driver.Event.Event's) have been cleared.
--
-- See "Foreign.CUDA.Driver.Event" and "Foreign.CUDA.Driver.Stream" for
-- more information on this topic.
--
--------------------------------------------------------------------------------

module Foreign.CUDA.Driver (

  module Foreign.CUDA.Ptr,
  module Foreign.CUDA.Driver.Context,
  module Foreign.CUDA.Driver.Device,
  module Foreign.CUDA.Driver.Error,
  module Foreign.CUDA.Driver.Exec,
  module Foreign.CUDA.Driver.Marshal,
  module Foreign.CUDA.Driver.Module,
  module Foreign.CUDA.Driver.Unified,
  module Foreign.CUDA.Driver.Utils

) where

-- If we don't import everything from a module below, the generated haddocks
-- will not have the convenient link to a module, but instead enumerate
-- documentation for everything which was imported. This, unfortunately, looks
-- like garbage, so it would be nice to find a better solution.
--    -- TLM 2018-11-18
--
import Foreign.CUDA.Ptr
import Foreign.CUDA.Driver.Context                        hiding ( device )
import Foreign.CUDA.Driver.Device
import Foreign.CUDA.Driver.Error
import Foreign.CUDA.Driver.Exec
import Foreign.CUDA.Driver.Marshal
import Foreign.CUDA.Driver.Module
import Foreign.CUDA.Driver.Unified
import Foreign.CUDA.Driver.Utils

