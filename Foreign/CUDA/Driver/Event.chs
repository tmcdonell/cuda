{-# LANGUAGE BangPatterns             #-}
{-# LANGUAGE CPP                      #-}
{-# LANGUAGE EmptyDataDecls           #-}
{-# LANGUAGE ForeignFunctionInterface #-}
--------------------------------------------------------------------------------
-- |
-- Module    : Foreign.CUDA.Driver.Event
-- Copyright : [2009..2014] Trevor L. McDonell
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
import Foreign.CUDA.Types
import Foreign.CUDA.Internal.C2HS
import Foreign.CUDA.Driver.Error

-- System
import Foreign
import Foreign.C
import Data.Maybe
import Control.Monad                                    ( liftM )
import Control.Exception                                ( throwIO )


--------------------------------------------------------------------------------
-- Event management
--------------------------------------------------------------------------------

-- |
-- Create a new event
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
{-# INLINEABLE destroy #-}
destroy :: Event -> IO ()
destroy !ev = nothingIfOk =<< cuEventDestroy ev

{-# INLINE cuEventDestroy #-}
{# fun unsafe cuEventDestroy
  { useEvent `Event' } -> `Status' cToEnum #}


-- |
-- Determine the elapsed time (in milliseconds) between two events
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
-- different device contexts. Requires cuda-3.2.
--
{-# INLINEABLE wait #-}
wait :: Event -> Maybe Stream -> [WaitFlag] -> IO ()
#if CUDA_VERSION < 3020
wait _ _ _           = requireSDK 3.2 "wait"
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
{-# INLINEABLE block #-}
block :: Event -> IO ()
block !ev = nothingIfOk =<< cuEventSynchronize ev

{-# INLINE cuEventSynchronize #-}
{# fun cuEventSynchronize
  { useEvent `Event' } -> `Status' cToEnum #}

