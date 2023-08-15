{-# LANGUAGE BangPatterns             #-}
{-# LANGUAGE CPP                      #-}
{-# LANGUAGE EmptyDataDecls           #-}
{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE TemplateHaskell          #-}
#ifdef USE_EMPTY_CASE
{-# LANGUAGE EmptyCase                #-}
#endif
--------------------------------------------------------------------------------
-- |
-- Module    : Foreign.CUDA.Driver.Event
-- Copyright : [2009..2023] Trevor L. McDonell
-- License   : BSD
--
-- Event management for low-level driver interface
--
--------------------------------------------------------------------------------

module Foreign.CUDA.Driver.Event (

  -- * Event Management
  Event(..), EventFlag(..), WaitFlag,
  create, destroy, elapsedTime, query, record, wait, block

) where

#include "cbits/stubs.h"
{# context lib="cuda" #}

-- Friends
import Foreign.CUDA.Internal.C2HS
import Foreign.CUDA.Driver.Error
import Foreign.CUDA.Driver.Stream                         ( Stream(..), defaultStream )

-- System
import Foreign
import Foreign.C
import Data.Maybe
import Control.Monad                                      ( liftM )
import Control.Exception                                  ( throwIO )


--------------------------------------------------------------------------------
-- Data Types
--------------------------------------------------------------------------------

-- |
-- Events are markers that can be inserted into the CUDA execution stream and
-- later queried.
--
newtype Event = Event { useEvent :: {# type CUevent #}}
  deriving (Eq, Show)

-- |
-- Event creation flags
--
{# enum CUevent_flags as EventFlag
    { underscoreToCase
    }
    with prefix="CU_EVENT" deriving (Eq, Show, Bounded) #}

-- |
-- Possible option flags for waiting for events
--
data WaitFlag
instance Enum WaitFlag where
#ifdef USE_EMPTY_CASE
  toEnum   x = error ("WaitFlag.toEnum: Cannot match " ++ show x)
  fromEnum x = case x of {}
#endif


--------------------------------------------------------------------------------
-- Event management
--------------------------------------------------------------------------------

-- |
-- Create a new event
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html#group__CUDA__EVENT_1g450687e75f3ff992fe01662a43d9d3db>
--
{-# INLINEABLE create #-}
create :: [EventFlag] -> IO Event
create !flags = resultIfOk =<< cuEventCreate flags

{-# INLINE cuEventCreate #-}
{# fun unsafe cuEventCreate
  { alloca-         `Event'       peekEvt*
  , combineBitMasks `[EventFlag]'          } -> `Status' cToEnum #}
  where peekEvt = liftM Event . peek


-- |
-- Destroy an event
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html#group__CUDA__EVENT_1g593ec73a8ec5a5fc031311d3e4dca1ef>
--
{-# INLINEABLE destroy #-}
destroy :: Event -> IO ()
destroy !ev = nothingIfOk =<< cuEventDestroy ev

{-# INLINE cuEventDestroy #-}
{# fun unsafe cuEventDestroy
  { useEvent `Event' } -> `Status' cToEnum #}


-- |
-- Determine the elapsed time (in milliseconds) between two events
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html#group__CUDA__EVENT_1gdfb1178807353bbcaa9e245da497cf97>
--
{-# INLINEABLE elapsedTime #-}
elapsedTime :: Event -> Event -> IO Float
elapsedTime !ev1 !ev2 = resultIfOk =<< cuEventElapsedTime ev1 ev2

{-# INLINE cuEventElapsedTime #-}
{# fun unsafe cuEventElapsedTime
  { alloca-  `Float' peekFloatConv*
  , useEvent `Event'
  , useEvent `Event'                } -> `Status' cToEnum #}


-- |
-- Determines if a event has actually been recorded
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html#group__CUDA__EVENT_1g6f0704d755066b0ee705749ae911deef>
--
{-# INLINEABLE query #-}
query :: Event -> IO Bool
query !ev =
  cuEventQuery ev >>= \rv ->
  case rv of
    Success  -> return True
    NotReady -> return False
    _        -> throwIO (ExitCode rv)

{-# INLINE cuEventQuery #-}
{# fun unsafe cuEventQuery
  { useEvent `Event' } -> `Status' cToEnum #}


-- |
-- Record an event once all operations in the current context (or optionally
-- specified stream) have completed. This operation is asynchronous.
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html#group__CUDA__EVENT_1g95424d3be52c4eb95d83861b70fb89d1>
--
{-# INLINEABLE record #-}
record :: Event -> Maybe Stream -> IO ()
record !ev !mst =
  nothingIfOk =<< cuEventRecord ev (fromMaybe defaultStream mst)

{-# INLINE cuEventRecord #-}
{# fun unsafe cuEventRecord
  { useEvent  `Event'
  , useStream `Stream' } -> `Status' cToEnum #}


-- |
-- Makes all future work submitted to the (optional) stream wait until the given
-- event reports completion before beginning execution. Synchronisation is
-- performed on the device, including when the event and stream are from
-- different device contexts.
--
-- Requires CUDA-3.2.
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html#group__CUDA__STREAM_1g6a898b652dfc6aa1d5c8d97062618b2f>
--
{-# INLINEABLE wait #-}
wait :: Event -> Maybe Stream -> [WaitFlag] -> IO ()
#if CUDA_VERSION < 3020
wait _ _ _           = requireSDK 'wait 3.2
#else
wait !ev !mst !flags =
  nothingIfOk =<< cuStreamWaitEvent (fromMaybe defaultStream mst) ev flags

{-# INLINE cuStreamWaitEvent #-}
{# fun unsafe cuStreamWaitEvent
  { useStream       `Stream'
  , useEvent        `Event'
  , combineBitMasks `[WaitFlag]' } -> `Status' cToEnum #}
#endif

-- |
-- Wait until the event has been recorded
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EVENT.html#group__CUDA__EVENT_1g9e520d34e51af7f5375610bca4add99c>
--
{-# INLINEABLE block #-}
block :: Event -> IO ()
block !ev = nothingIfOk =<< cuEventSynchronize ev

{-# INLINE cuEventSynchronize #-}
{# fun cuEventSynchronize
  { useEvent `Event' } -> `Status' cToEnum #}

