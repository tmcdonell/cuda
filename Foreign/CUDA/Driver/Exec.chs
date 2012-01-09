{-# LANGUAGE GADTs, CPP, ForeignFunctionInterface, EmptyDataDecls #-}
--------------------------------------------------------------------------------
-- |
-- Module    : Foreign.CUDA.Driver.Exec
-- Copyright : (c) [2009..2011] Trevor L. McDonell
-- License   : BSD
--
-- Kernel execution control for low-level driver interface
--
--------------------------------------------------------------------------------

module Foreign.CUDA.Driver.Exec (

  -- * Kernel Execution
  Fun(Fun), FunParam(..), FunAttribute(..), CacheConfig(..),
  requires, setBlockShape, setSharedSize, setParams, setCacheConfigFun,
  launch, launchKernel, launchKernel'

) where

#include <cuda.h>
{# context lib="cuda" #}

-- Friends
import Foreign.CUDA.Internal.C2HS
import Foreign.CUDA.Driver.Error
import Foreign.CUDA.Driver.Stream               (Stream(..))
import Foreign.CUDA.Driver.Texture              (Texture(..))

-- System
import Foreign
import Foreign.C
import Data.Maybe
import Control.Monad                            (zipWithM_)


#if CUDA_VERSION >= 3020
{-# DEPRECATED TArg "as of CUDA version 3.2" #-}
#endif
#if CUDA_VERSION >= 4000
{-# DEPRECATED setBlockShape, setSharedSize, setParams, launch
      "use launchKernel instead" #-}
#endif


--------------------------------------------------------------------------------
-- Data Types
--------------------------------------------------------------------------------

-- |
-- A @__global__@ device function
--
newtype Fun = Fun { useFun :: {# type CUfunction #}}


-- |
-- Function attributes
--
{# enum CUfunction_attribute as FunAttribute
    { underscoreToCase
    , MAX_THREADS_PER_BLOCK as MaxKernelThreadsPerBlock }
    with prefix="CU_FUNC_ATTRIBUTE" deriving (Eq, Show) #}

-- |
-- Cache configuration preference
--
#if CUDA_VERSION < 3000
data CacheConfig
#else
{# enum CUfunc_cache_enum as CacheConfig
    { underscoreToCase }
    with prefix="CU_FUNC_CACHE_PREFER" deriving (Eq, Show) #}
#endif

-- |
-- Kernel function parameters
--
data FunParam where
  IArg :: !Int             -> FunParam
  FArg :: !Float           -> FunParam
  TArg :: !Texture         -> FunParam
  VArg :: Storable a => !a -> FunParam

instance Storable FunParam where
  sizeOf (IArg _)       = sizeOf (undefined :: CUInt)
  sizeOf (FArg _)       = sizeOf (undefined :: CFloat)
  sizeOf (VArg v)       = sizeOf v
  sizeOf (TArg _)       = 0

  alignment (IArg _)    = alignment (undefined :: CUInt)
  alignment (FArg _)    = alignment (undefined :: CFloat)
  alignment (VArg v)    = alignment v
  alignment (TArg _)    = 0

  poke p (IArg i)       = poke (castPtr p) i
  poke p (FArg f)       = poke (castPtr p) f
  poke p (VArg v)       = poke (castPtr p) v
  poke _ (TArg _)       = return ()


--------------------------------------------------------------------------------
-- Execution Control
--------------------------------------------------------------------------------

-- |
-- Returns the value of the selected attribute requirement for the given kernel
--
requires :: Fun -> FunAttribute -> IO Int
requires fn att = resultIfOk =<< cuFuncGetAttribute att fn

{# fun unsafe cuFuncGetAttribute
  { alloca-   `Int'          peekIntConv*
  , cFromEnum `FunAttribute'
  , useFun    `Fun'                       } -> `Status' cToEnum #}


-- |
-- Specify the @(x,y,z)@ dimensions of the thread blocks that are created when
-- the given kernel function is launched.
--
setBlockShape :: Fun -> (Int,Int,Int) -> IO ()
setBlockShape fn (x,y,z) = nothingIfOk =<< cuFuncSetBlockShape fn x y z

{# fun unsafe cuFuncSetBlockShape
  { useFun `Fun'
  ,        `Int'
  ,        `Int'
  ,        `Int' } -> `Status' cToEnum #}


-- |
-- Set the number of bytes of dynamic shared memory to be available to each
-- thread block when the function is launched
--
setSharedSize :: Fun -> Integer -> IO ()
setSharedSize fn bytes = nothingIfOk =<< cuFuncSetSharedSize fn bytes

{# fun unsafe cuFuncSetSharedSize
  { useFun   `Fun'
  , cIntConv `Integer' } -> `Status' cToEnum #}


-- |
-- On devices where the L1 cache and shared memory use the same hardware
-- resources, this sets the preferred cache configuration for the given device
-- function. This is only a preference; the driver is free to choose a different
-- configuration as required to execute the function.
--
-- Switching between configuration modes may insert a device-side
-- synchronisation point for streamed kernel launches.
--
setCacheConfigFun :: Fun -> CacheConfig -> IO ()
#if CUDA_VERSION < 3000
setCacheConfigFun _  _    = requireSDK 3.0 "setCacheConfigFun"
#else
setCacheConfigFun fn pref = nothingIfOk =<< cuFuncSetCacheConfig fn pref

{# fun unsafe cuFuncSetCacheConfig
  { useFun    `Fun'
  , cFromEnum `CacheConfig' } -> `Status' cToEnum #}
#endif

-- |
-- Invoke the kernel on a size @(w,h)@ grid of blocks. Each block contains the
-- number of threads specified by a previous call to 'setBlockShape'. The launch
-- may also be associated with a specific 'Stream'.
--
launch :: Fun -> (Int,Int) -> Maybe Stream -> IO ()
launch fn (w,h) mst =
  nothingIfOk =<< case mst of
    Nothing -> cuLaunchGridAsync fn w h (Stream nullPtr)
    Just st -> cuLaunchGridAsync fn w h st

{# fun unsafe cuLaunchGridAsync
  { useFun    `Fun'
  ,           `Int'
  ,           `Int'
  , useStream `Stream' } -> `Status' cToEnum #}


-- |
-- Invoke a kernel on a @(gx * gy * gz)@ grid of blocks, where each block
-- contains @(tx * ty * tz)@ threads and has access to a given number of bytes
-- of shared memory. The launch may also be associated with a specific 'Stream'.
--
-- In 'launchKernel', the number of kernel parameters and their offsets and
-- sizes do not need to be specified, as this information is retrieved directly
-- from the kernel's image. This requires the kernel to have been compiled with
-- toolchain version 3.2 or later.
--
-- The alternative 'launchKernel'' will pass the arguments in directly,
-- requiring the application to know the size and alignment/padding of each
-- kernel parameter.
--
launchKernel, launchKernel'
    :: Fun                      -- ^ function to execute
    -> (Int,Int,Int)            -- ^ block grid dimension
    -> (Int,Int,Int)            -- ^ thread block shape
    -> Int                      -- ^ shared memory (bytes)
    -> Maybe Stream             -- ^ (optional) stream to execute in
    -> [FunParam]               -- ^ list of function parameters
    -> IO ()
#if CUDA_VERSION >= 4000
launchKernel fn (gx,gy,gz) (tx,ty,tz) sm mst args
  = (=<<) nothingIfOk
  $ withMany withFP args
  $ \pa -> withArray pa
  $ \pp -> cuLaunchKernel fn gx gy gz tx ty tz sm st pp nullPtr
  where
    st = fromMaybe (Stream nullPtr) mst

    withFP :: FunParam -> (Ptr FunParam -> IO b) -> IO b
    withFP p f = case p of
      IArg v -> with' v (f . castPtr)
      FArg v -> with' v (f . castPtr)
      VArg v -> with' v (f . castPtr)
      TArg _ -> error "launchKernel: TArg is deprecated"

    -- can't use the standard 'with' because 'alloca' will pass an undefined
    -- dummy argument when determining 'sizeOf' and 'alignment', but sometimes
    -- instances in Accelerate need to evaluate this argument.
    --
    with' :: Storable a => a -> (Ptr a -> IO b) -> IO b
    with' val f =
      allocaBytes (sizeOf val) $ \ptr -> do
        poke ptr val
        f ptr


launchKernel' fn (gx,gy,gz) (tx,ty,tz) sm mst args
  = (=<<) nothingIfOk
  $ with bytes
  $ \pb -> withArray' args
  $ \pa -> withArray0 nullPtr [buffer, castPtr pa, size, castPtr pb]
  $ \pp -> cuLaunchKernel fn gx gy gz tx ty tz sm st nullPtr pp
  where
    buffer      = wordPtrToPtr 0x01     -- CU_LAUNCH_PARAM_BUFFER_POINTER
    size        = wordPtrToPtr 0x02     -- CU_LAUNCH_PARAM_BUFFER_SIZE
    bytes       = foldl (\a x -> a + sizeOf x) 0 args
    st          = fromMaybe (Stream nullPtr) mst

    -- can't use the standard 'withArray' because 'mallocArray' will pass
    -- 'undefined' to 'sizeOf' when determining how many bytes to allocate, but
    -- our Storable instance for FunParam needs to dispatch on each constructor,
    -- hence evaluating the undefined.
    --
    withArray' vals f =
      allocaBytes bytes $ \ptr -> do
        pokeArray ptr vals
        f ptr


{# fun unsafe cuLaunchKernel
  { useFun    `Fun'
  ,           `Int', `Int', `Int'
  ,           `Int', `Int', `Int'
  ,           `Int'
  , useStream `Stream'
  , castPtr   `Ptr (Ptr FunParam)'
  , castPtr   `Ptr (Ptr ())'       } -> `Status' cToEnum #}

#else
launchKernel fn (gx,gy,_) (tx,ty,tz) sm mst args = do
  setParams     fn args
  setSharedSize fn (toInteger sm)
  setBlockShape fn (tx,ty,tz)
  launch        fn (gx,gy) mst

launchKernel' = launchKernel
#endif


--------------------------------------------------------------------------------
-- Kernel function parameters
--------------------------------------------------------------------------------

-- |
-- Set the parameters that will specified next time the kernel is invoked
--
setParams :: Fun -> [FunParam] -> IO ()
setParams fn prs = do
  zipWithM_ (set fn) offsets prs
  nothingIfOk =<< cuParamSetSize fn (last offsets)
  where
    offsets = scanl (\a b -> a + size b) 0 prs

    size (IArg _)    = sizeOf (undefined :: CUInt)
    size (FArg _)    = sizeOf (undefined :: CFloat)
    size (TArg _)    = 0
    size (VArg v)    = sizeOf v

    set f o (IArg v) = nothingIfOk =<< cuParamSeti f o v
    set f o (FArg v) = nothingIfOk =<< cuParamSetf f o v
    set f _ (TArg v) = nothingIfOk =<< cuParamSetTexRef f (-1) v
    set f o (VArg v) = with v $ \p -> (nothingIfOk =<< cuParamSetv f o p (sizeOf v))


{# fun unsafe cuParamSetSize
  { useFun `Fun'
  ,        `Int' } -> `Status' cToEnum #}

{# fun unsafe cuParamSeti
  { useFun `Fun'
  ,        `Int'
  ,        `Int' } -> `Status' cToEnum #}

{# fun unsafe cuParamSetf
  { useFun `Fun'
  ,        `Int'
  ,        `Float' } -> `Status' cToEnum #}

{# fun unsafe cuParamSetv
  `Storable a' =>
  { useFun  `Fun'
  ,         `Int'
  , castPtr `Ptr a'
  ,         `Int'   } -> `Status' cToEnum #}

{# fun unsafe cuParamSetTexRef
  { useFun     `Fun'
  ,            `Int'    -- must be CU_PARAM_TR_DEFAULT (-1)
  , useTexture `Texture' } -> `Status' cToEnum #}

