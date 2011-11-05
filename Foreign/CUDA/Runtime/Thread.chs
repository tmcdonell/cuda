--------------------------------------------------------------------------------
-- |
-- Module    : Foreign.CUDA.Runtime.Thread
-- Copyright : (c) [2009..2011] Trevor L. McDonell
-- License   : BSD
--
-- Thread management routines
--
--------------------------------------------------------------------------------

module Foreign.CUDA.Runtime.Thread
  {-# DEPRECATED "superseded by functions in Device module" #-} (

  -- * Cache configuration
  D.Limit(..),
  getLimit, setLimit,

  -- * Thread management
  sync, exit

) where

import qualified Foreign.CUDA.Runtime.Device      as D

--------------------------------------------------------------------------------
-- Thread management
--------------------------------------------------------------------------------

-- |
-- Block until the device has completed all preceding requested tasks. Returns
-- an error if one of the tasks fails.
--
sync :: IO ()
sync = D.sync
#if CUDART_VERSION >= 4000
{-# DEPRECATED exit "use 'Foreign.CUDA.Runtime.Device.sync' instead" #-}
#endif


-- |
-- Explicitly clean up all runtime related resources associated with the calling
-- host thread. Any subsequent API call re-initialised the environment.
-- Implicitly called on thread exit.
--
exit :: IO ()
exit = D.reset
#if CUDART_VERSION >= 4000
{-# DEPRECATED exit "use 'Foreign.CUDA.Runtime.Device.reset' instead" #-}
#endif


-- |
-- Query compute 2.0 call stack limits. Requires cuda-3.1.
--
getLimit :: D.Limit -> IO Int
getLimit = D.getLimit
#if CUDART_VERSION >= 4000
{-# DEPRECATED exit "use 'Foreign.CUDA.Runtime.Device.getLimit' instead" #-}
#endif


-- |
-- Set compute 2.0 call stack limits. Requires cuda-3.1.
--
setLimit :: D.Limit -> Int -> IO ()
setLimit = D.setLimit
#if CUDART_VERSION >= 4000
{-# DEPRECATED exit "use 'Foreign.CUDA.Runtime.Device.setLimit' instead" #-}
#endif

