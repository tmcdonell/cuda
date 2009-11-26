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
    Fun, Module,
    getFun, loadFile, loadData, loadDataEx, unload
  )
  where

#include <cuda.h>
{# context lib="cuda" #}

-- Friends
import Foreign.CUDA.Driver.Error
import Foreign.CUDA.Internal.C2HS

-- System
import Foreign
import Foreign.C
import Foreign.ForeignPtr

import Data.ByteString (ByteString, useAsCString)


--------------------------------------------------------------------------------
-- Data Types
--------------------------------------------------------------------------------

--
-- Function
--
{# pointer *CUfunction as Fun foreign newtype #}
withFun :: Fun -> (Ptr Fun -> IO a) -> IO a

newFun :: IO Fun
newFun = Fun `fmap` mallocForeignPtrBytes (sizeOf (undefined :: Ptr ()))


--
-- Module
--
{# pointer *CUmodule as Module foreign newtype #}
withModule :: Module -> (Ptr Module -> IO a) -> IO a

newModule :: IO Module
newModule = Module `fmap` mallocForeignPtrBytes (sizeOf (undefined :: Ptr ()))

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

--
-- Returns a function handle
--
getFun :: Module -> String -> IO (Either String Fun)
getFun modu fname =
  newFun          >>= \fun -> withFun fun $ \fp  ->
  withModule modu $   \mp  ->
  cuModuleGetFunction fp mp fname >>= \rv ->
    return $ case nothingIfOk rv of
      Nothing -> Right fun
      Just e  -> Left e

{# fun unsafe cuModuleGetFunction
  { id           `Ptr Fun'
  , castPtr      `Ptr Module'
  , withCString* `String'     } -> `Status' cToEnum #}


--
-- Load the contents of the specified file (either a ptx or cubin file) to
-- create a new module, and load that module into the current context
--
loadFile :: String -> IO (Either String Module)
loadFile ptx =
  newModule           >>= \m  -> withModule m $ \mp ->
  cuModuleLoad mp ptx >>= \rv ->
    return $ case nothingIfOk rv of
      Nothing -> Right m
      Just e  -> Left e

{# fun unsafe cuModuleLoad
  { id           `Ptr Module'
  , withCString* `String'     } -> `Status' cToEnum #}


--
-- Load the contents of the given image into a new module, and load that module
-- into the current context. The image (typically) is the contents of a cubin or
-- ptx file as a NULL-terminated string.
--
loadData :: ByteString -> IO (Either String Module)
loadData img =
  newModule               >>= \m  -> withModule m $ \mp ->
  cuModuleLoadData mp img >>= \rv ->
    return $ case nothingIfOk rv of
      Nothing -> Right m
      Just e  -> Left e

{# fun unsafe cuModuleLoadData
  { id     `Ptr Module'
  , useBS* `ByteString' } -> ` Status' cToEnum #}
  where useBS bs act = useAsCString bs $ \p -> act (castPtr p)


--
-- Load a module with online compiler options. Would also need to unpack those
-- options which also return values.
--
loadDataEx :: ByteString -> [JITOption] -> IO (Either String (Module, [JITOption]))
loadDataEx = error "Foreign.CUDA.Driver.Module.loadDataEx: not implemented yet"


--
-- Unload a module from the current context
--
unload :: Module -> IO (Maybe String)
unload m = withModule m $ \mp -> (nothingIfOk `fmap` cuModuleUnload mp)

{# fun unsafe cuModuleUnload
  { castPtr `Ptr Module' } -> `Status' cToEnum #}

