{-# LANGUAGE BangPatterns             #-}
{-# LANGUAGE ForeignFunctionInterface #-}
--------------------------------------------------------------------------------
-- |
-- Module    : Foreign.CUDA.Driver.Module.Query
-- Copyright : [2009..2014] Trevor L. McDonell
-- License   : BSD
--
-- Querying module attributes for low-level driver interface
--
--------------------------------------------------------------------------------

module Foreign.CUDA.Driver.Module.Query (

  -- ** Querying module inhabitants
  getFun, getPtr, getTex,

) where

#include "cbits/stubs.h"
{# context lib="cuda" #}

-- Friends
import Foreign.CUDA.Driver.Error
import Foreign.CUDA.Driver.Exec
import Foreign.CUDA.Driver.Marshal                      ( peekDeviceHandle )
import Foreign.CUDA.Driver.Module.Base
import Foreign.CUDA.Driver.Texture
import Foreign.CUDA.Internal.C2HS
import Foreign.CUDA.Ptr

-- System
import Foreign
import Foreign.C
import Control.Exception                                ( throwIO )
import Control.Monad                                    ( liftM )


--------------------------------------------------------------------------------
-- Querying module attributes
--------------------------------------------------------------------------------

-- |
-- Returns a function handle.
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html#group__CUDA__MODULE_1ga52be009b0d4045811b30c965e1cb2cf>
--
{-# INLINEABLE getFun #-}
getFun :: Module -> String -> IO Fun
getFun !mdl !fn = resultIfFound "function" fn =<< cuModuleGetFunction mdl fn

{-# INLINE cuModuleGetFunction #-}
{# fun unsafe cuModuleGetFunction
  { alloca-      `Fun'    peekFun*
  , useModule    `Module'
  , withCString* `String'          } -> `Status' cToEnum #}
  where peekFun = liftM Fun . peek


-- |
-- Return a global pointer, and size of the global (in bytes).
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html#group__CUDA__MODULE_1gf3e43672e26073b1081476dbf47a86ab>
--
{-# INLINEABLE getPtr #-}
getPtr :: Module -> String -> IO (DevicePtr a, Int)
getPtr !mdl !name = do
  (!status,!dptr,!bytes) <- cuModuleGetGlobal mdl name
  resultIfFound "global" name (status,(dptr,bytes))

{-# INLINE cuModuleGetGlobal #-}
{# fun unsafe cuModuleGetGlobal
  { alloca-      `DevicePtr a' peekDeviceHandle*
  , alloca-      `Int'         peekIntConv*
  , useModule    `Module'
  , withCString* `String'                        } -> `Status' cToEnum #}


-- |
-- Return a handle to a texture reference. This texture reference handle
-- should not be destroyed, as the texture will be destroyed automatically
-- when the module is unloaded.
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html#group__CUDA__MODULE_1g9607dcbf911c16420d5264273f2b5608>
--
{-# INLINEABLE getTex #-}
getTex :: Module -> String -> IO Texture
getTex !mdl !name = resultIfFound "texture" name =<< cuModuleGetTexRef mdl name

{-# INLINE cuModuleGetTexRef #-}
{# fun unsafe cuModuleGetTexRef
  { alloca-      `Texture' peekTex*
  , useModule    `Module'
  , withCString* `String'           } -> `Status' cToEnum #}


--------------------------------------------------------------------------------
-- Internal
--------------------------------------------------------------------------------

{-# INLINE resultIfFound #-}
resultIfFound :: String -> String -> (Status, a) -> IO a
resultIfFound kind name (!status,!result) =
  case status of
       Success  -> return result
       NotFound -> cudaError (kind ++ ' ' : describe status ++ ": " ++ name)
       _        -> throwIO (ExitCode status)

