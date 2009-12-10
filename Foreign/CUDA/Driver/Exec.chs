{-# LANGUAGE ForeignFunctionInterface #-}
--------------------------------------------------------------------------------
-- |
-- Module    : Foreign.CUDA.Driver.Exec
-- Copyright : (c) 2009 Trevor L. McDonell
-- License   : BSD
--
-- Kernel execution control for low-level driver interface
--
--------------------------------------------------------------------------------

module Foreign.CUDA.Driver.Exec
  (
    Fun, FunParam(..), FunAttribute(..), newFun, withFun,
    requires, setBlockShape, setSharedSize, setParams, launch
  )
  where

#include <cuda.h>
{# context lib="cuda" #}

-- Friends
import Foreign.CUDA.Internal.C2HS
import Foreign.CUDA.Driver.Error
import Foreign.CUDA.Driver.Stream               (Stream, withStream)

-- System
import Foreign
import Foreign.C
import Data.Maybe
import Control.Monad                            (zipWithM)


--------------------------------------------------------------------------------
-- Data Types
--------------------------------------------------------------------------------

-- |
-- A __global__ device function
--
{# pointer *CUfunction as Fun foreign newtype #}
withFun :: Fun -> (Ptr Fun -> IO a) -> IO a

newFun :: IO Fun
newFun = Fun `fmap` mallocForeignPtrBytes (sizeOf (undefined :: Ptr ()))

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
data Storable a => FunParam a
    = IArg Int
    | FArg Float
    | VArg a
--  | TRef Texture

--------------------------------------------------------------------------------
-- Execution Control
--------------------------------------------------------------------------------

-- |
-- Returns the value of the selected attribute requirement for the given kernel
--
requires :: Fun -> FunAttribute -> IO (Either String Int)
requires fn att = withFun fn $ \f -> (resultIfOk `fmap` cuFuncGetAttribute att f)

{# fun unsafe cuFuncGetAttribute
  { alloca-   `Int'          peekIntConv*
  , cFromEnum `FunAttribute'
  , castPtr   `Ptr Fun'                   } -> `Status' cToEnum #}


-- |
-- Specify the (x,y,z) dimensions of the thread blocks that are created when the
-- given kernel function is lanched
--
setBlockShape :: Fun -> (Int,Int,Int) -> IO (Maybe String)
setBlockShape fn (x,y,z) = withFun fn $ \f -> (nothingIfOk `fmap` cuFuncSetBlockShape f x y z)

{# fun unsafe cuFuncSetBlockShape
  { castPtr `Ptr Fun'
  ,         `Int'
  ,         `Int'
  ,         `Int'     } -> `Status' cToEnum #}


-- |
-- Set the number of bytes of dynamic shared memory to be available to each
-- thread block when the function is launched
--
setSharedSize :: Fun -> Integer -> IO (Maybe String)
setSharedSize fn bytes = withFun fn $ \f -> (nothingIfOk `fmap` cuFuncSetSharedSize f bytes)

{# fun unsafe cuFuncSetSharedSize
  { castPtr  `Ptr Fun'
  , cIntConv `Integer' } -> `Status' cToEnum #}


-- |
-- Invoke the kernel on a size (w,h) grid of blocks. Each block contains the
-- number of threads specified by a previous call to `setBlockShape'. The launch
-- may also be associated with a specific `Stream'.
--
launch :: Fun -> (Int,Int) -> Maybe Stream -> IO (Maybe String)
launch fn (w,h) mst =
  withFun fn $ \f ->
  nothingIfOk `fmap` case mst of
    Nothing -> cuLaunchGridAsync f w h nullPtr
    Just st -> (withStream st $ \s -> cuLaunchGridAsync f w h s)

{# fun unsafe cuLaunchGridAsync
  { castPtr `Ptr Fun'
  ,         `Int'
  ,         `Int'
  , castPtr `Ptr Stream' } -> `Status' cToEnum #}


--------------------------------------------------------------------------------
-- Kernel function parameters
--------------------------------------------------------------------------------

-- |
-- Set the parameters that will specified next time the kernel is invoked
--
setParams :: Storable a => Fun -> [FunParam a] -> IO (Maybe [String])
setParams fn prs =
  withFun fn $ \f ->
  cuParamSetSize f (last offsets) >>= \rv ->
  case nothingIfOk rv of
    Just err -> return (Just [err])
    Nothing  -> (maybeNull . catMaybes) `fmap` zipWithM (set f) offsets prs
  where
    maybeNull [] = Nothing
    maybeNull x  = Just x

    offsets = scanl (\a b -> a + size b) 0 prs

    size (IArg v)    = sizeOf v
    size (FArg v)    = sizeOf v
    size (VArg v)    = sizeOf v

    set f o (IArg v) = nothingIfOk `fmap` cuParamSeti f o v
    set f o (FArg v) = nothingIfOk `fmap` cuParamSetf f o v
    set f o (VArg v) = with v $ \p -> (nothingIfOk `fmap` cuParamSetv f o p (sizeOf v))


{# fun unsafe cuParamSetSize
  { castPtr `Ptr Fun'
  ,         `Int'     } -> `Status' cToEnum #}

{# fun unsafe cuParamSeti
  { castPtr `Ptr Fun'
  ,         `Int'
  ,         `Int'     } -> `Status' cToEnum #}

{# fun unsafe cuParamSetf
  { castPtr `Ptr Fun'
  ,         `Int'
  ,         `Float'   } -> `Status' cToEnum #}

{# fun unsafe cuParamSetv
  `Storable a' =>
  { castPtr `Ptr Fun'
  ,         `Int'
  , castPtr `Ptr a'
  ,         `Int'     } -> `Status' cToEnum #}

