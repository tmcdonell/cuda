{-# LANGUAGE ForeignFunctionInterface #-}
--------------------------------------------------------------------------------
-- |
-- Module    : Foreign.CUDA.Runtime.Exec
-- Copyright : (c) 2009 Trevor L. McDonell
-- License   : BSD
--
-- Kernel execution control for C-for-CUDA runtime interface
--
--------------------------------------------------------------------------------

module Foreign.CUDA.Runtime.Exec
  (
    FunAttributes(..),
    attributes, setConfig, setParams, launch
  )
  where

#include "cbits/stubs.h"
#include <cuda_runtime_api.h>
{# context lib="cudart" #}

-- Friends
import Foreign.CUDA.Runtime.Stream
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

--
-- Function Attributes
--
{# pointer *cudaFuncAttributes as ^ foreign -> FunAttributes nocode #}

data FunAttributes = FunAttributes
  {
    constSizeBytes           :: Int64,
    localSizeBytes           :: Int64,
    sharedSizeBytes          :: Int64,
    maxKernelThreadsPerBlock :: Int,	-- ^ maximum block size that can be successively launched (based on register usage)
    numRegs                  :: Int     -- ^ number of registers required for each thread
  }
  deriving (Show)

instance Storable FunAttributes where
  sizeOf _    = {# sizeof cudaFuncAttributes #}
  alignment _ = alignment (undefined :: Ptr ())

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


--------------------------------------------------------------------------------
-- Execution Control
--------------------------------------------------------------------------------

-- |
-- Obtain the attributes of the named @__global__@ device function. This
-- itemises the requirements to successfully launch the given kernel.
--
attributes :: String -> IO FunAttributes
attributes fn = resultIfOk =<< cudaFuncGetAttributes fn

{# fun unsafe cudaFuncGetAttributes
  { alloca-      `FunAttributes' peek*
  , withCString* `String'              } -> `Status' cToEnum #}


-- |
-- Specify the grid and block dimensions for a device call. Used in conjunction
-- with 'setParams', this pushes data onto the execution stack that will be
-- popped when a function is 'launch'ed.
--
setConfig :: (Int,Int)		-- ^ grid dimensions
	  -> (Int,Int,Int)	-- ^ block dimensions
	  -> Int64		-- ^ shared memory per block (bytes)
	  -> Maybe Stream	-- ^ associated processing stream
	  -> IO ()
setConfig (gx,gy) (bx,by,bz) sharedMem mst =
  nothingIfOk =<< case mst of
    Nothing -> cudaConfigureCallSimple gx gy bx by bz sharedMem (Stream 0)
    Just st -> cudaConfigureCallSimple gx gy bx by bz sharedMem st

--
-- The FFI does not support passing deferenced structures to C functions, as
-- this is highly platform/compiler dependent. Wrap our own function stub
-- accepting plain integers.
--
{# fun unsafe cudaConfigureCallSimple
  {	      `Int', `Int'
  ,           `Int', `Int', `Int'
  , cIntConv  `Int64'
  , useStream `Stream'            } -> `Status' cToEnum #}


-- |
-- Set the argument parameters that will be passed to the next kernel
-- invocation. This is used in conjunction with 'setConfig' to control kernel
-- execution.
setParams :: Storable a => [a] -> IO ()
setParams prs =
  zipWithM_ set prs offsets
  where
    set v o = nothingIfOk =<< cudaSetupArgument v (sizeOf v) o
    offsets = scanl (\a b -> a + sizeOf b) 0 prs

{# fun unsafe cudaSetupArgument
  `Storable a' =>
  { with'* `a'
  ,        `Int'
  ,        `Int'   } -> `Status' cToEnum #}
  where
    with' v a = with v $ \p -> a (castPtr p)


-- |
-- Invoke the named kernel on the device, which must have been declared
-- @__global__@. This must be preceded by a call to 'setConfig' and (if
-- appropriate) 'setParams'.
--
launch :: String -> IO ()
launch fn = nothingIfOk =<< cudaLaunch fn

{# fun unsafe cudaLaunch
  { withCString* `String' } -> `Status' cToEnum #}

