{-# LANGUAGE BangPatterns             #-}
{-# LANGUAGE CPP                      #-}
{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE TemplateHaskell          #-}
--------------------------------------------------------------------------------
-- |
-- Module    : Foreign.CUDA.Driver.Graph.Exec
-- Copyright : [2018] Trevor L. McDonell
-- License   : BSD
--
-- Graph execution functions for the low-level driver interface
--
-- Requires CUDA-10
--
--------------------------------------------------------------------------------

module Foreign.CUDA.Driver.Graph.Exec (

  Executable(..),

  -- ** Execution
  launch,
  instantiate,
  destroy,

) where

#include "cbits/stubs.h"
{# context lib="cuda" #}

import Foreign.CUDA.Driver.Error
import Foreign.CUDA.Driver.Graph.Base
import Foreign.CUDA.Driver.Stream                         ( Stream(..) )
import Foreign.CUDA.Internal.C2HS

import Foreign
import Foreign.C

import Control.Monad                                      ( liftM )
import Data.ByteString.Char8                              ( ByteString )
import qualified Data.ByteString.Char8                    as B
import qualified Data.ByteString.Internal                 as B


--------------------------------------------------------------------------------
-- Graph execution
--------------------------------------------------------------------------------

-- | Execute a graph in the given stream. Only one instance may execute at
-- a time; to execute a graph concurrently, it must be 'instantiate'd into
-- multiple executables.
--
-- Requires CUDA-10.0
--
-- <https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g6b2dceb3901e71a390d2bd8b0491e471>
--
-- @since 0.10.0.0
--
{-# INLINEABLE launch #-}
#if CUDA_VERSION < 10000
launch :: Executable -> Stream -> IO ()
launch _ _ = requireSDK 'launch 10.0
#else
{# fun unsafe cuGraphLaunch as launch
  { useExecutable `Executable'
  , useStream     `Stream'
  }
  -> `()' checkStatus*- #}
#endif


-- | Instantiate the task graph description of a program into an executable
-- graph.
--
-- Requires CUDA-10.0
--
-- <https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g433ae118a751c9f2087f53d7add7bc2c>
--
-- @since 0.10.0.0
--
{-# INLINEABLE instantiate #-}
instantiate :: Graph -> IO Executable
#if CUDA_VERSION < 10000
instantiate _ = requireSDK 'instantiate 10.0
#else
instantiate !g = do
  let logSize = 2048
  allocaArray logSize $ \p_elog -> do
    (s, e, n) <- cuGraphInstantiate g p_elog logSize
    --
    case s of
      Success -> return e
      _       -> do
        errLog <- peekCStringLen (p_elog, logSize)
        cudaErrorIO (unlines [describe s, "phErrorNode = " ++ show n, errLog])

{-# INLINE cuGraphInstantiate #-}
{# fun unsafe cuGraphInstantiate
  { alloca-  `Executable' peekExecutable*
  , useGraph `Graph'
  , alloca-  `Maybe Node' peekErrNode*
  , castPtr  `CString'
  ,          `Int'
  }
  -> `Status' cToEnum #}
  where
    peekExecutable  = liftM Executable . peek
    peekErrNode p   = if p == nullPtr
                        then return Nothing
                        else liftM (Just . Node) (peek p)
#endif


-- | Destroy an executable graph
--
-- Requires CUDA-10.0
--
-- <https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1ga32ad4944cc5d408158207c978bc43a7>
--
-- @since 0.10.0.0
--
{-# INLINEABLE destroy #-}
#if CUDA_VERSION < 10000
destroy :: Executable -> IO ()
destroy _ = requireSDK 'destroy 10.0
#else
{# fun unsafe cuGraphExecDestroy as destroy
  { useExecutable `Executable'
  }
  -> `()' checkStatus*- #}
#endif

