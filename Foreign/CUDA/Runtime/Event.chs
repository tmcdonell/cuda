{-# LANGUAGE BangPatterns             #-}
{-# LANGUAGE CPP                      #-}
{-# LANGUAGE EmptyCase                #-}
{-# LANGUAGE EmptyDataDecls           #-}
{-# LANGUAGE ForeignFunctionInterface #-}
--------------------------------------------------------------------------------
-- |
-- Module    : Foreign.CUDA.Driver.Event
-- Copyright : [2009..2014] Trevor L. McDonell
-- License   : BSD
--
-- Event management for C-for-CUDA runtime environment
--
--------------------------------------------------------------------------------

module Foreign.CUDA.Runtime.Event (

  -- * Event Management
  Event, EventFlag(..), WaitFlag,
  create, destroy, elapsedTime, query, record, wait, block

) where

#include "cbits/stubs.h"
{# context lib="cudart" #}

-- Friends
import Foreign.CUDA.Runtime.Error
import Foreign.CUDA.Runtime.Stream                      ( Stream(..), defaultStream )
import Foreign.CUDA.Internal.C2HS

-- System
import Foreign
import Foreign.C
import Control.Monad                                    ( liftM )
import Control.Exception                                ( throwIO )
import Data.Maybe                                       ( fromMaybe )


#c
typedef enum cudaEvent_option_enum {
//  CUDA_EVENT_OPTION_DEFAULT       = cuduEventDefault,
    CUDA_EVENT_OPTION_BLOCKING_SYNC = cudaEventBlockingSync
} cudaEvent_option;
#endc

--------------------------------------------------------------------------------
-- Data Types
--------------------------------------------------------------------------------

-- |
-- Events
--
newtype Event = Event { useEvent :: {# type cudaEvent_t #}}
  deriving (Eq, Show)

-- |
-- Event creation flags
--
{# enum cudaEvent_option_enum as EventFlag
    { underscoreToCase }
    with prefix="CUDA_EVENT_OPTION" deriving (Eq,Show) #}

-- |
-- Possible option flags for waiting for events
--
data WaitFlag
instance Enum WaitFlag where
  toEnum   x = case x of {}
  fromEnum x = case x of {}


--------------------------------------------------------------------------------
-- Event management
--------------------------------------------------------------------------------

-- |
-- Create a new event
--
{-# INLINEABLE create #-}
create :: [EventFlag] -> IO Event
create !flags = resultIfOk =<< cudaEventCreateWithFlags flags

{-# INLINE cudaEventCreateWithFlags #-}
{# fun unsafe cudaEventCreateWithFlags
  { alloca-         `Event'       peekEvt*
  , combineBitMasks `[EventFlag]'          } -> `Status' cToEnum #}
  where peekEvt = liftM Event . peek


-- |
-- Destroy an event
--
{-# INLINEABLE destroy #-}
destroy :: Event -> IO ()
destroy !ev = nothingIfOk =<< cudaEventDestroy ev

{-# INLINE cudaEventDestroy #-}
{# fun unsafe cudaEventDestroy
  { useEvent `Event' } -> `Status' cToEnum #}


-- |
-- Determine the elapsed time (in milliseconds) between two events
--
{-# INLINEABLE elapsedTime #-}
elapsedTime :: Event -> Event -> IO Float
elapsedTime !ev1 !ev2 = resultIfOk =<< cudaEventElapsedTime ev1 ev2

{-# INLINE cudaEventElapsedTime #-}
{# fun unsafe cudaEventElapsedTime
  { alloca-  `Float' peekFloatConv*
  , useEvent `Event'
  , useEvent `Event'                } -> `Status' cToEnum #}


-- |
-- Determines if a event has actually been recorded
--
{-# INLINEABLE query #-}
query :: Event -> IO Bool
query !ev =
  cudaEventQuery ev >>= \rv ->
  case rv of
    Success  -> return True
    NotReady -> return False
    _        -> throwIO (ExitCode rv)

{-# INLINE cudaEventQuery #-}
{# fun unsafe cudaEventQuery
  { useEvent `Event' } -> `Status' cToEnum #}


-- |
-- Record an event once all operations in the current context (or optionally
-- specified stream) have completed. This operation is asynchronous.
--
{-# INLINEABLE record #-}
record :: Event -> Maybe Stream -> IO ()
record !ev !mst =
  nothingIfOk =<< cudaEventRecord ev (maybe defaultStream id mst)

{-# INLINE cudaEventRecord #-}
{# fun unsafe cudaEventRecord
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
#if CUDART_VERSION < 3020
wait _ _ _           = requireSDK 3.2 "wait"
#else
wait !ev !mst !flags =
  let st = fromMaybe defaultStream mst
  in  nothingIfOk =<< cudaStreamWaitEvent st ev flags

{-# INLINE cudaStreamWaitEvent #-}
{# fun unsafe cudaStreamWaitEvent
  { useStream       `Stream'
  , useEvent        `Event'
  , combineBitMasks `[WaitFlag]' } -> `Status' cToEnum #}
#endif

-- |
-- Wait until the event has been recorded
--
{-# INLINEABLE block #-}
block :: Event -> IO ()
block !ev = nothingIfOk =<< cudaEventSynchronize ev

{-# INLINE cudaEventSynchronize #-}
{# fun cudaEventSynchronize
  { useEvent `Event' } -> `Status' cToEnum #}

