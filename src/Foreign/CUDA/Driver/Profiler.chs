{-# LANGUAGE ForeignFunctionInterface #-}
--------------------------------------------------------------------------------
-- |
-- Module    : Foreign.CUDA.Driver.Profiler
-- Copyright : [2009..2017] Trevor L. McDonell
-- License   : BSD
--
-- Profiler control for low-level driver interface
--
--------------------------------------------------------------------------------

module Foreign.CUDA.Driver.Profiler (

  OutputMode(..),
  initialise,
  start, stop,

) where

#include "cbits/stubs.h"
{# context lib="cuda" #}

-- friends
import Foreign.CUDA.Driver.Error
import Foreign.CUDA.Internal.C2HS

-- system
import Foreign
import Foreign.C


-- | Profiler output mode
--
{# enum CUoutput_mode as OutputMode
    { underscoreToCase
    , CU_OUT_CSV as CSV }
    with prefix="CU_OUT" deriving (Eq, Show) #}


-- | Initialise the CUDA profiler.
--
-- The configuration file is used to specify profiling options and profiling
-- counters. Refer to the "Compute Command Line Profiler User Guide" for
-- supported profiler options and counters.
--
-- Note that the CUDA profiler can not be initialised with this function if
-- another profiling tool is already active.
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PROFILER.html#group__CUDA__PROFILER>
--
{-# INLINEABLE initialise #-}
initialise
    :: FilePath     -- ^ configuration file that itemises which counters and/or options to profile
    -> FilePath     -- ^ output file where profiling results will be stored
    -> OutputMode
    -> IO ()
initialise config output mode
  = nothingIfOk =<< cuProfilerInitialize config output mode

{-# INLINE cuProfilerInitialize #-}
{# fun unsafe cuProfilerInitialize
  {           `String'
  ,           `String'
  , cFromEnum `OutputMode'
  }
  -> `Status' cToEnum #}


-- | Begin profiling collection by the active profiling tool for the current
-- context. If profiling is already enabled, then this has no effect.
--
-- 'start' and 'stop' can be used to programatically control profiling
-- granularity, by allowing profiling to be done only on selected pieces of
-- code.
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PROFILER.html#group__CUDA__PROFILER_1g8a5314de2292c2efac83ac7fcfa9190e>
--
{-# INLINEABLE start #-}
start :: IO ()
start = nothingIfOk =<< cuProfilerStart

{-# INLINE cuProfilerStart #-}
{# fun unsafe cuProfilerStart
  { } -> `Status' cToEnum #}


-- | Stop profiling collection by the active profiling tool for the current
-- context, and force all pending profiler events to be written to the output
-- file. If profiling is already inactive, this has no effect.
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PROFILER.html#group__CUDA__PROFILER_1g4d8edef6174fd90165e6ac838f320a5f>
--
{-# INLINEABLE stop #-}
stop :: IO ()
stop = nothingIfOk =<< cuProfilerStop

{-# INLINE cuProfilerStop #-}
{# fun unsafe cuProfilerStop
  { } -> `Status' cToEnum #}

