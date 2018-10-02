{-# LANGUAGE BangPatterns             #-}
{-# LANGUAGE CPP                      #-}
{-# LANGUAGE EmptyDataDecls           #-}
{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE MagicHash                #-}
{-# LANGUAGE TemplateHaskell          #-}
--------------------------------------------------------------------------------
-- |
-- Module    : Foreign.CUDA.Driver.Module.Link
-- Copyright : [2009..2018] Trevor L. McDonell
-- License   : BSD
--
-- Module linking for low-level driver interface
--
-- Since CUDA-5.5
--
--------------------------------------------------------------------------------

module Foreign.CUDA.Driver.Module.Link (

  -- ** JIT module linking
  LinkState, JITOption(..), JITInputType(..),

  create, destroy, complete,
  addFile,
  addData, addDataFromPtr,

) where

#include "cbits/stubs.h"
{# context lib="cuda" #}

-- Friends
import Foreign.CUDA.Driver.Error
import Foreign.CUDA.Driver.Module.Base
import Foreign.CUDA.Internal.C2HS

-- System
import Control.Monad                                    ( liftM )
import Foreign
import Foreign.C
import Unsafe.Coerce

import GHC.Ptr
import Data.ByteString.Char8                            ( ByteString )
import qualified Data.ByteString.Char8                  as B


--------------------------------------------------------------------------------
-- Data Types
--------------------------------------------------------------------------------

-- |
-- A pending JIT linker state
--
#if CUDA_VERSION < 5050
data LinkState
#else
newtype LinkState = LinkState { useLinkState :: {# type CUlinkState #} }
  deriving (Show)
#endif


--------------------------------------------------------------------------------
-- JIT linking
--------------------------------------------------------------------------------

-- |
-- Create a pending JIT linker invocation. The returned 'LinkState' should
-- be 'destroy'ed once no longer needed. The device code machine size will
-- match the calling application.
--
-- Requires CUDA-5.5.
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html#group__CUDA__MODULE_1g86ca4052a2fab369cb943523908aa80d>
--
{-# INLINEABLE create #-}
create :: [JITOption] -> IO LinkState
#if CUDA_VERSION < 5050
create _        = requireSDK 'create 5.5
#else
create !options =
  let (opt,val) = unzip $ map jitOptionUnpack options
  in
  withArray (map cFromEnum opt)    $ \p_opts ->
  withArray (map unsafeCoerce val) $ \p_vals ->
    resultIfOk =<< cuLinkCreate (length opt) p_opts p_vals

{-# INLINE cuLinkCreate #-}
{# fun unsafe cuLinkCreate
  {         `Int'
  , id      `Ptr CInt'
  , id      `Ptr (Ptr ())'
  , alloca- `LinkState'    peekLS* } -> `Status' cToEnum #}
  where
    peekLS = liftM LinkState . peek
#endif


-- |
-- Destroy the state of a JIT linker invocation.
--
-- Requires CUDA-5.5.
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html#group__CUDA__MODULE_1g01b7ae2a34047b05716969af245ce2d9>
--
{-# INLINEABLE destroy #-}
destroy :: LinkState -> IO ()
#if CUDA_VERSION < 5050
destroy _  = requireSDK 'destroy 5.5
#else
destroy !s = nothingIfOk =<< cuLinkDestroy s

{-# INLINE cuLinkDestroy #-}
{# fun unsafe cuLinkDestroy
  { useLinkState `LinkState' } -> `Status' cToEnum #}
#endif


-- |
-- Complete a pending linker invocation and load the current module. The
-- link state will be destroyed.
--
-- Requires CUDA-5.5.
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html#group__CUDA__MODULE_1g818fcd84a4150a997c0bba76fef4e716>
--
{-# INLINEABLE complete #-}
complete :: LinkState -> IO Module
#if CUDA_VERSION < 5050
complete _   = requireSDK 'complete 5.5
#else
complete !ls = do
  cubin <- resultIfOk =<< cuLinkComplete ls nullPtr
  mdl   <- loadDataFromPtr (castPtr cubin)
  destroy ls
  return mdl

{-# INLINE cuLinkComplete #-}
{# fun unsafe cuLinkComplete
  { useLinkState `LinkState'
  , alloca-      `Ptr ()'    peek*
  , castPtr      `Ptr Int'
  }
  -> `Status' cToEnum #}
#endif


-- |
-- Add an input file to a pending linker invocation.
--
-- Requires CUDA-5.5.
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html#group__CUDA__MODULE_1g1224c0fd48d4a683f3ce19997f200a8c>
--
{-# INLINEABLE addFile #-}
addFile :: LinkState -> FilePath -> JITInputType -> [JITOption] -> IO ()
#if CUDA_VERSION < 5050
addFile _ _ _ _ = requireSDK 'addFile 5.5
#else
addFile !ls !fp !t !options =
  let (opt,val) = unzip $ map jitOptionUnpack options
  in
  withArrayLen (map cFromEnum opt)    $ \i p_opts ->
  withArray    (map unsafeCoerce val) $ \  p_vals ->
    nothingIfOk =<< cuLinkAddFile ls t fp i p_opts p_vals

{-# INLINE cuLinkAddFile #-}
{# fun unsafe cuLinkAddFile
  { useLinkState `LinkState'
  , cFromEnum    `JITInputType'
  , withCString* `FilePath'
  ,              `Int'
  , id           `Ptr CInt'
  , id           `Ptr (Ptr ())'
  }
  -> `Status' cToEnum #}
#endif


-- |
-- Add an input to a pending linker invocation.
--
-- Requires CUDA-5.5.
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html#group__CUDA__MODULE_1g3ebcd2ccb772ba9c120937a2d2831b77>
--
{-# INLINEABLE addData #-}
addData :: LinkState -> ByteString -> JITInputType -> [JITOption] -> IO ()
#if CUDA_VERSION < 5050
addData _ _ _ = requireSDK 'addData 5.5
#else
addData !ls !img !k !options =
  B.useAsCStringLen img (\(p, n) -> addDataFromPtr ls n (castPtr p) k options)
#endif

-- |
-- As 'addData', but read the specified number of bytes of image data from
-- the given pointer.
--
{-# INLINEABLE addDataFromPtr #-}
addDataFromPtr :: LinkState -> Int -> Ptr Word8 -> JITInputType -> [JITOption] -> IO ()
#if CUDA_VERSION < 5050
addDataFromPtr _ _ _ _ = requireSDK 'addDataFromPtr 5.5
#else
addDataFromPtr !ls !n !img !t !options =
  let (opt,val) = unzip $ map jitOptionUnpack options
      name      = Ptr "<unknown>"#
  in
  withArrayLen (map cFromEnum opt)    $ \i p_opts ->
  withArray    (map unsafeCoerce val) $ \  p_vals ->
    nothingIfOk =<< cuLinkAddData ls t img n name i p_opts p_vals

{-# INLINE cuLinkAddData #-}
{# fun unsafe cuLinkAddData
  { useLinkState `LinkState'
  , cFromEnum    `JITInputType'
  , castPtr      `Ptr Word8'
  ,              `Int'
  , id           `Ptr CChar'
  ,              `Int'
  , id           `Ptr CInt'
  , id           `Ptr (Ptr ())'
  }
  -> `Status' cToEnum #}
#endif

