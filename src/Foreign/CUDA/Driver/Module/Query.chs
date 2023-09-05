{-# LANGUAGE BangPatterns             #-}
{-# LANGUAGE CPP                      #-}
{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE MagicHash                #-}
{-# LANGUAGE UnboxedTuples            #-}
--------------------------------------------------------------------------------
-- |
-- Module    : Foreign.CUDA.Driver.Module.Query
-- Copyright : [2009..2023] Trevor L. McDonell
-- License   : BSD
--
-- Querying module attributes for low-level driver interface
--
--------------------------------------------------------------------------------

module Foreign.CUDA.Driver.Module.Query (

  -- ** Querying module inhabitants
  getFun, getPtr,

) where

#include "cbits/stubs.h"
{# context lib="cuda" #}

-- Friends
import Foreign.CUDA.Driver.Error
import Foreign.CUDA.Driver.Exec
import Foreign.CUDA.Driver.Marshal                      ( peekDeviceHandle )
import Foreign.CUDA.Driver.Module.Base
import Foreign.CUDA.Internal.C2HS
import Foreign.CUDA.Ptr

-- System
import Foreign
import Foreign.C
import Control.Exception                                ( throwIO )
import Control.Monad                                    ( liftM )
import Data.ByteString.Short                            ( ShortByteString )
import qualified Data.ByteString.Short                  as BS
import qualified Data.ByteString.Short.Internal         as BI
import qualified Data.ByteString.Internal               as BI
import Prelude                                          as P

import GHC.Exts
import GHC.Base                                         ( IO(..) )


--------------------------------------------------------------------------------
-- Querying module attributes
--------------------------------------------------------------------------------

-- |
-- Returns a function handle.
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html#group__CUDA__MODULE_1ga52be009b0d4045811b30c965e1cb2cf>
--
{-# INLINEABLE getFun #-}
getFun :: Module -> ShortByteString -> IO Fun
getFun !mdl !fn = resultIfFound "function" fn =<< cuModuleGetFunction mdl fn

{-# INLINE cuModuleGetFunction #-}
{# fun unsafe cuModuleGetFunction
  { alloca-       `Fun'             peekFun*
  , useModule     `Module'
  , useAsCString* `ShortByteString'
  }
  -> `Status' cToEnum #}
  where
    peekFun = liftM Fun . peek


-- |
-- Return a global pointer, and size of the global (in bytes).
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html#group__CUDA__MODULE_1gf3e43672e26073b1081476dbf47a86ab>
--
{-# INLINEABLE getPtr #-}
getPtr :: Module -> ShortByteString -> IO (DevicePtr a, Int)
getPtr !mdl !name = do
  (!status,!dptr,!bytes) <- cuModuleGetGlobal mdl name
  resultIfFound "global" name (status,(dptr,bytes))

{-# INLINE cuModuleGetGlobal #-}
{# fun unsafe cuModuleGetGlobal
  { alloca-       `DevicePtr a'     peekDeviceHandle*
  , alloca-       `Int'             peekIntConv*
  , useModule     `Module'
  , useAsCString* `ShortByteString'
  }
  -> `Status' cToEnum #}


--------------------------------------------------------------------------------
-- Internal
--------------------------------------------------------------------------------

{-# INLINE resultIfFound #-}
resultIfFound :: String -> ShortByteString -> (Status, a) -> IO a
resultIfFound kind name (!status,!result) =
  case status of
       Success  -> return result
       NotFound -> cudaErrorIO (kind ++ ' ' : describe status ++ ": " ++ unpack name)
       _        -> throwIO (ExitCode status)


-- Utilities
-- ---------

-- [Short]ByteStrings are not null-terminated, so can't be passed directly to C.
--
-- unsafeUseAsCString :: ShortByteString -> CString
-- unsafeUseAsCString (BI.SBS ba#) = Ptr (byteArrayContents# ba#)

{-# INLINE useAsCString #-}
useAsCString :: ShortByteString -> (CString -> IO a) -> IO a
useAsCString (BI.SBS ba#) action = IO $ \s0 ->
  case sizeofByteArray# ba#                              of { n# ->
  case newPinnedByteArray# (n# +# 1#) s0                 of { (# s1, mba# #) ->
  case byteArrayContents# (unsafeCoerce# mba#)           of { addr# ->
  case copyByteArrayToAddr# ba# 0# addr# n# s1           of { s2 ->
  case writeWord8OffAddr# addr# n# (wordToWord8# 0##) s2 of { s3 ->
  case action (Ptr addr#)                                of { IO action' ->
  case action' s3                                        of { (# s4, r  #) ->
  case touch# mba# s4                                    of { s5 ->
  (# s5, r #)
 }}}}}}}}


{-# INLINE unpack #-}
unpack :: ShortByteString -> [Char]
unpack = P.map BI.w2c . BS.unpack

#if __GLASGOW_HASKELL__ < 902
{-# INLINE wordToWord8# #-}
wordToWord8# :: Word# -> Word#
wordToWord8# x = x
#endif

