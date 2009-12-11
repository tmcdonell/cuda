{-# LANGUAGE ForeignFunctionInterface #-}
--------------------------------------------------------------------------------
-- |
-- Module    : Foreign.CUDA.Driver.Module
-- Copyright : (c) 2009 Trevor L. McDonell
-- License   : BSD
--
-- Module management for low-level driver interface
--
--------------------------------------------------------------------------------

module Foreign.CUDA.Driver.Module
  (
    Module,
    getFun, loadFile, loadData, loadDataEx, unload
  )
  where

#include <cuda.h>
{# context lib="cuda" #}

-- Friends
import Foreign.CUDA.Internal.C2HS
import Foreign.CUDA.Driver.Error
import Foreign.CUDA.Driver.Exec

-- System
import Foreign
import Foreign.C

import Control.Monad                            (liftM)
import Data.ByteString                          (ByteString, useAsCString)


--------------------------------------------------------------------------------
-- Data Types
--------------------------------------------------------------------------------

-- |
-- A reference to a Module object, containing collections of device functions
--
newtype Module = Module { useModule :: {# type CUmodule #}}
  deriving (Show)

instance Storable Module where
  sizeOf _    = sizeOf (undefined :: {# type CUmodule #})
  alignment _ = alignment (undefined :: {# type CUmodule #})
  peek p      = Module `fmap` peek (castPtr p)
  poke p v    = poke (castPtr p) (useModule v)


--
-- Just-in-time compilation
-- XXX: Actually, probably want to build an algebraic data type manually, so
-- that we can associate the acceptable value type to each option.
--
{# enum CUjit_option as JITOption
    { underscoreToCase }
    with prefix="CU_JIT" deriving (Eq, Show) #}

{# enum CUjit_target as JITTarget
    { underscoreToCase }
    with prefix="CU_TARGET" deriving (Eq, Show) #}

{# enum CUjit_fallback as JITFallback
    { underscoreToCase }
    with prefix="CU_PREFER" deriving (Eq, Show) #}


--------------------------------------------------------------------------------
-- Module management
--------------------------------------------------------------------------------

-- |
-- Returns a function handle
--
getFun :: Module -> String -> IO (Either String Fun)
getFun mdl fn = resultIfOk `fmap` cuModuleGetFunction mdl fn

{# fun unsafe cuModuleGetFunction
  { alloca-      `Fun'    peekFun*
  , useModule    `Module'
  , withCString* `String'          } -> `Status' cToEnum #}
  where peekFun = liftM Fun . peek


-- |
-- Load the contents of the specified file (either a ptx or cubin file) to
-- create a new module, and load that module into the current context
--
loadFile :: String -> IO (Either String Module)
loadFile ptx = resultIfOk `fmap` cuModuleLoad ptx

{# fun unsafe cuModuleLoad
  { alloca-      `Module' peekMod*
  , withCString* `String'          } -> `Status' cToEnum #}
  where peekMod = liftM Module . peek


-- |
-- Load the contents of the given image into a new module, and load that module
-- into the current context. The image (typically) is the contents of a cubin or
-- ptx file as a NULL-terminated string.
--
loadData :: ByteString -> IO (Either String Module)
loadData img = resultIfOk `fmap` cuModuleLoadData img

{# fun unsafe cuModuleLoadData
  { alloca- `Module'     peekMod*
  , useBS*  `ByteString'          } -> ` Status' cToEnum #}
  where
    peekMod      = liftM Module . peek
    useBS bs act = useAsCString bs $ \p -> act (castPtr p)


-- |
-- Load a module with online compiler options. Would also need to unpack those
-- options which also return values.
--
loadDataEx :: ByteString -> [JITOption] -> IO (Either String (Module, [JITOption]))
loadDataEx = error "Foreign.CUDA.Driver.Module.loadDataEx: not implemented yet"


-- |
-- Unload a module from the current context
--
unload :: Module -> IO (Maybe String)
unload m = nothingIfOk `fmap` cuModuleUnload m

{# fun unsafe cuModuleUnload
  { useModule `Module' } -> `Status' cToEnum #}

