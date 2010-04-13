{-# LANGUAGE GADTs #-}
{-# LANGUAGE ForeignFunctionInterface #-}
--------------------------------------------------------------------------------
-- |
-- Module    : Foreign.CUDA.Driver.Exec
-- Copyright : (c) [2009..2010] Trevor L. McDonell
-- License   : BSD
--
-- Kernel execution control for low-level driver interface
--
--------------------------------------------------------------------------------

module Foreign.CUDA.Driver.Exec
  (
    Fun(Fun),  -- need to export the data constructor for use by Module )=
    FunParam(..), FunAttribute(..),
    requires, setBlockShape, setSharedSize, setParams, launch
  )
  where

#include <cuda.h>
{# context lib="cuda" #}

-- Friends
import Foreign.CUDA.Internal.C2HS
import Foreign.CUDA.Driver.Error
import Foreign.CUDA.Driver.Stream               (Stream(..))

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


-- |
-- Kernel function parameters
--
data FunParam where
  IArg :: Int   -> FunParam
  FArg :: Float -> FunParam
--  TArg :: Texture -> FunParam
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
    size (VArg v)    = sizeOf v

    set f o (IArg v) = nothingIfOk =<< cuParamSeti f o v
    set f o (FArg v) = nothingIfOk =<< cuParamSetf f o v
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

