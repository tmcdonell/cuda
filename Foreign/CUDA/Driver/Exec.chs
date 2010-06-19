{-# LANGUAGE GADTs, ForeignFunctionInterface #-}
--------------------------------------------------------------------------------
-- |
-- Module    : Foreign.CUDA.Driver.Exec
-- Copyright : (c) [2009..2010] Trevor L. McDonell
-- License   : BSD
--
-- Kernel execution control for low-level driver interface
--
--------------------------------------------------------------------------------

#include <cuda.h>
{# context lib="cuda" #}

module Foreign.CUDA.Driver.Exec
  (
    Fun(Fun), FunParam(..), FunAttribute(..),
#if CUDA_VERSION >= 3000
    CacheConfig(..),
#endif
    requires, setBlockShape, setSharedSize, setParams,
#if CUDA_VERSION >= 3000
    setCacheConfig,
#endif
    launch
  )
  where

-- Friends
import Foreign.CUDA.Internal.C2HS
import Foreign.CUDA.Driver.Error
import Foreign.CUDA.Driver.Stream               (Stream(..))
import Foreign.CUDA.Driver.Texture              (Texture(..))

-- System
import Foreign
import Foreign.C
import Control.Monad                            (zipWithM_)


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

#if CUDA_VERSION >= 3000
-- |
-- Cache configuration preference
--
{# enum CUfunc_cache_enum as CacheConfig
    { underscoreToCase }
    with prefix="CU_FUNC_CACHE_PREFER" deriving (Eq, Show) #}
#endif

-- |
-- Kernel function parameters
--
data FunParam where
  IArg :: Int   -> FunParam
  FArg :: Float -> FunParam
  TArg :: Texture a -> FunParam
  VArg :: Storable a => a -> FunParam


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
-- Specify the (x,y,z) dimensions of the thread blocks that are created when the
-- given kernel function is lanched
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

#if CUDA_VERSION >= 3000
-- |
-- On devices where the L1 cache and shared memory use the same hardware
-- resources, this sets the preferred cache configuration for the given device
-- function. This is only a preference; the driver is free to choose a different
-- configuration as required to execute the function.
--
-- Switching between configuration modes may insert a device-side
-- synchronisation point for streamed kernel launches.
--
setCacheConfig :: Fun -> CacheConfig -> IO ()
setCacheConfig fn pref = nothingIfOk =<< cuFuncSetCacheConfig fn pref

{# fun unsafe cuFuncSetCacheConfig
  { useFun    `Fun'
  , cFromEnum `CacheConfig' } -> `Status' cToEnum #}
#endif

-- |
-- Invoke the kernel on a size (w,h) grid of blocks. Each block contains the
-- number of threads specified by a previous call to `setBlockShape'. The launch
-- may also be associated with a specific `Stream'.
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

    size (IArg _)    = sizeOf (undefined::CUInt)
    size (FArg _)    = sizeOf (undefined::CFloat)
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
  , useTexture `Texture a' } -> `Status' cToEnum #}

