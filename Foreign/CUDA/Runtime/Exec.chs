{-# LANGUAGE BangPatterns             #-}
{-# LANGUAGE CPP                      #-}
{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE GADTs                    #-}
--------------------------------------------------------------------------------
-- |
-- Module    : Foreign.CUDA.Runtime.Exec
-- Copyright : [2009..2014] Trevor L. McDonell
-- License   : BSD
--
-- Kernel execution control for C-for-CUDA runtime interface
--
--------------------------------------------------------------------------------

module Foreign.CUDA.Runtime.Exec (

  -- * Kernel Execution
  Fun, FunAttributes(..), FunParam(..), CacheConfig(..),
  attributes, setConfig, setParams, setCacheConfig, launch, launchKernel,

) where

#include "cbits/stubs.h"
{# context lib="cudart" #}

-- Friends
import Foreign.CUDA.Runtime.Stream                      ( Stream(..), defaultStream )
import Foreign.CUDA.Runtime.Error
import Foreign.CUDA.Internal.C2HS

-- System
import Foreign
import Foreign.C
import Control.Monad

#c
typedef struct cudaFuncAttributes cudaFuncAttributes;
#endc


--------------------------------------------------------------------------------
-- Data Types
--------------------------------------------------------------------------------

-- |
-- A @__global__@ device function.
--
-- Note that the use of a string naming a function was deprecated in CUDA 4.1
-- and removed in CUDA 5.0.
--
#if CUDART_VERSION >= 5000
type Fun = FunPtr ()
#else
type Fun = String
#endif


--
-- Function Attributes
--
{# pointer *cudaFuncAttributes as ^ foreign -> FunAttributes nocode #}

data FunAttributes = FunAttributes
  {
    constSizeBytes           :: !Int64,
    localSizeBytes           :: !Int64,
    sharedSizeBytes          :: !Int64,
    maxKernelThreadsPerBlock :: !Int,   -- ^ maximum block size that can be successively launched (based on register usage)
    numRegs                  :: !Int    -- ^ number of registers required for each thread
  }
  deriving (Show)

instance Storable FunAttributes where
  sizeOf _    = {# sizeof cudaFuncAttributes #}
  alignment _ = alignment (undefined :: Ptr ())

  poke _ _    = error "Can not poke Foreign.CUDA.Runtime.FunAttributes"
  peek p      = do
    cs <- cIntConv `fmap` {#get cudaFuncAttributes.constSizeBytes#} p
    ls <- cIntConv `fmap` {#get cudaFuncAttributes.localSizeBytes#} p
    ss <- cIntConv `fmap` {#get cudaFuncAttributes.sharedSizeBytes#} p
    tb <- cIntConv `fmap` {#get cudaFuncAttributes.maxThreadsPerBlock#} p
    nr <- cIntConv `fmap` {#get cudaFuncAttributes.numRegs#} p

    return FunAttributes
      {
        constSizeBytes           = cs,
        localSizeBytes           = ls,
        sharedSizeBytes          = ss,
        maxKernelThreadsPerBlock = tb,
        numRegs                  = nr
      }

#if CUDART_VERSION < 3000
data CacheConfig
#else
-- |
-- Cache configuration preference
--
{# enum cudaFuncCache as CacheConfig
    { }
    with prefix="cudaFuncCachePrefer" deriving (Eq, Show) #}
#endif

-- |
-- Kernel function parameters. Doubles will be converted to an internal float
-- representation on devices that do not support doubles natively.
--
data FunParam where
  IArg :: !Int             -> FunParam
  FArg :: !Float           -> FunParam
  DArg :: !Double          -> FunParam
  VArg :: Storable a => !a -> FunParam


--------------------------------------------------------------------------------
-- Execution Control
--------------------------------------------------------------------------------

-- |
-- Obtain the attributes of the named @__global__@ device function. This
-- itemises the requirements to successfully launch the given kernel.
--
{-# INLINEABLE attributes #-}
attributes :: Fun -> IO FunAttributes
attributes !fn = resultIfOk =<< cudaFuncGetAttributes fn

{-# INLINE cudaFuncGetAttributes #-}
{# fun unsafe cudaFuncGetAttributes
  { alloca-  `FunAttributes' peek*
  , withFun* `Fun'                 } -> `Status' cToEnum #}


-- |
-- Specify the grid and block dimensions for a device call. Used in conjunction
-- with 'setParams', this pushes data onto the execution stack that will be
-- popped when a function is 'launch'ed.
--
{-# INLINEABLE setConfig #-}
setConfig :: (Int,Int)          -- ^ grid dimensions
          -> (Int,Int,Int)      -- ^ block dimensions
          -> Int64              -- ^ shared memory per block (bytes)
          -> Maybe Stream       -- ^ associated processing stream
          -> IO ()
setConfig (!gx,!gy) (!bx,!by,!bz) !sharedMem !mst =
  nothingIfOk =<<
    cudaConfigureCallSimple gx gy bx by bz sharedMem (maybe defaultStream id mst)


--
-- The FFI does not support passing deferenced structures to C functions, as
-- this is highly platform/compiler dependent. Wrap our own function stub
-- accepting plain integers.
--
{-# INLINE cudaConfigureCallSimple #-}
{# fun unsafe cudaConfigureCallSimple
  {           `Int', `Int'
  ,           `Int', `Int', `Int'
  , cIntConv  `Int64'
  , useStream `Stream'            } -> `Status' cToEnum #}


-- |
-- Set the argument parameters that will be passed to the next kernel
-- invocation. This is used in conjunction with 'setConfig' to control kernel
-- execution.
--
{-# INLINEABLE setParams #-}
setParams :: [FunParam] -> IO ()
setParams = foldM_ k 0
  where
    k !offset !arg = do
      let s = size arg
      set arg s offset >>= nothingIfOk
      return (offset + s)

    size (IArg _) = sizeOf (undefined :: Int)
    size (FArg _) = sizeOf (undefined :: Float)
    size (DArg _) = sizeOf (undefined :: Double)
    size (VArg a) = sizeOf a

    set (IArg v) s o = cudaSetupArgument v s o
    set (FArg v) s o = cudaSetupArgument v s o
    set (VArg v) s o = cudaSetupArgument v s o
    set (DArg v) s o =
      cudaSetDoubleForDevice v >>= resultIfOk >>= \d ->
      cudaSetupArgument d s o


{-# INLINE cudaSetupArgument #-}
{# fun unsafe cudaSetupArgument
  `Storable a' =>
  { with'* `a'
  ,        `Int'
  ,        `Int'   } -> `Status' cToEnum #}
  where
    with' v a = with v $ \p -> a (castPtr p)

{-# INLINE cudaSetDoubleForDevice #-}
{# fun unsafe cudaSetDoubleForDevice
  { with'* `Double' peek'* } -> `Status' cToEnum #}
  where
    with' v a = with v $ \p -> a (castPtr p)
    peek'     = peek . castPtr


-- |
-- On devices where the L1 cache and shared memory use the same hardware
-- resources, this sets the preferred cache configuration for the given device
-- function. This is only a preference; the driver is free to choose a different
-- configuration as required to execute the function.
--
-- Switching between configuration modes may insert a device-side
-- synchronisation point for streamed kernel launches
--
{-# INLINEABLE setCacheConfig #-}
setCacheConfig :: Fun -> CacheConfig -> IO ()
#if CUDART_VERSION < 3000
setCacheConfig _ _       = requireSDK 3.0 "setCacheConfig"
#else
setCacheConfig !fn !pref = nothingIfOk =<< cudaFuncSetCacheConfig fn pref

{-# INLINE cudaFuncSetCacheConfig #-}
{# fun unsafe cudaFuncSetCacheConfig
  { withFun*  `Fun'
  , cFromEnum `CacheConfig' } -> `Status' cToEnum #}
#endif


-- |
-- Invoke the @__global__@ kernel function on the device. This must be preceded
-- by a call to 'setConfig' and (if appropriate) 'setParams'.
--
{-# INLINEABLE launch #-}
launch :: Fun -> IO ()
launch !fn = nothingIfOk =<< cudaLaunch fn

{-# INLINE cudaLaunch #-}
{# fun unsafe cudaLaunch
  { withFun* `Fun' } -> `Status' cToEnum #}


-- |
-- Invoke a kernel on a @(gx * gy)@ grid of blocks, where each block contains
-- @(tx * ty * tz)@ threads and has access to a given number of bytes of shared
-- memory. The launch may also be associated with a specific 'Stream'.
--
{-# INLINEABLE launchKernel #-}
launchKernel
    :: Fun              -- ^ Device function symbol
    -> (Int,Int)        -- ^ grid dimensions
    -> (Int,Int,Int)    -- ^ thread block shape
    -> Int64            -- ^ shared memory per block (bytes)
    -> Maybe Stream     -- ^ (optional) execution stream
    -> [FunParam]
    -> IO ()
launchKernel !fn !grid !block !sm !mst !args = do
  setConfig grid block sm mst
  setParams args
  launch fn

--------------------------------------------------------------------------------
-- Internals
--------------------------------------------------------------------------------

-- CUDA 5.0 changed the type of a kernel function from char* to void*
--
withFun :: Fun -> (Ptr a -> IO b) -> IO b
#if CUDART_VERSION >= 5000
withFun fn action = action (castFunPtrToPtr fn)
#else
withFun           = withCString
#endif

