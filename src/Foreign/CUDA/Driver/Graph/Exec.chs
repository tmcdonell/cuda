{-# LANGUAGE BangPatterns             #-}
{-# LANGUAGE CPP                      #-}
{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE TemplateHaskell          #-}
--------------------------------------------------------------------------------
-- |
-- Module    : Foreign.CUDA.Driver.Graph.Exec
-- Copyright : [2018..2020] Trevor L. McDonell
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
  setKernel,

) where

#include "cbits/stubs.h"
{# context lib="cuda" #}

import Foreign.CUDA.Driver.Error
import Foreign.CUDA.Driver.Exec                           ( Fun(..), FunParam(..) )
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
launch = requireSDK 'launch 10.0
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
instantiate = requireSDK 'instantiate 10.0
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


-- | Update the parameters for a kernel node in the given executable graph
--
-- Requires CUDA-10.1
--
-- <https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gd84243569e4c3d6356b9f2eea20ed48c>
--
-- @since 0.10.1.0
--
setKernel
    :: Executable
    -> Node
    -> Fun
    -> (Int, Int, Int)  -- ^ grid dimension
    -> (Int, Int, Int)  -- ^ thread block dimensions
    -> Int              -- ^ shared memory (bytes)
    -> [FunParam]
    -> IO ()
#if CUDA_VERSION < 10010
setKernel = requireSDK 'setKernel 10.1
#else
setKernel !exe !n !fun (!gx,!gy,!gz) (!tx,!ty,!tz) !sm !args
  = withMany withFP args
  $ \pa -> withArray pa
  $ \pp -> cuGraphExecKernelNodeSetParams_simple exe n fun gx gy gz tx ty tz sm pp
  where
    withFP :: FunParam -> (Ptr () -> IO b) -> IO b
    withFP !p !f = case p of
      IArg v -> with' v (f . castPtr)
      FArg v -> with' v (f . castPtr)
      VArg v -> with' v (f . castPtr)

    -- can't use the standard 'with' because 'alloca' will pass an undefined
    -- dummy argument when determining 'sizeOf' and 'alignment', but sometimes
    -- instances in Accelerate need to evaluate this argument.
    --
    with' :: Storable a => a -> (Ptr a -> IO b) -> IO b
    with' !val !f =
      allocaBytes (sizeOf val) $ \ptr -> do
        poke ptr val
        f ptr

    {# fun unsafe cuGraphExecKernelNodeSetParams_simple
      { useExecutable `Executable'
      , useNode       `Node'
      , useFun        `Fun'
      ,               `Int'
      ,               `Int'
      ,               `Int'
      ,               `Int'
      ,               `Int'
      ,               `Int'
      ,               `Int'
      , id            `Ptr (Ptr ())'
      }
      -> `()' checkStatus*- #}
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
destroy = requireSDK 'destroy 10.0
#else
{# fun unsafe cuGraphExecDestroy as destroy
  { useExecutable `Executable'
  }
  -> `()' checkStatus*- #}
#endif

