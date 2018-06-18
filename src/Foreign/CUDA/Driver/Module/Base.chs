{-# LANGUAGE BangPatterns             #-}
{-# LANGUAGE CPP                      #-}
{-# LANGUAGE EmptyDataDecls           #-}
{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE TemplateHaskell          #-}
{-# OPTIONS_HADDOCK prune #-}
--------------------------------------------------------------------------------
-- |
-- Module    : Foreign.CUDA.Driver.Module.Base
-- Copyright : [2009..2017] Trevor L. McDonell
-- License   : BSD
--
-- Module loading for low-level driver interface
--
--------------------------------------------------------------------------------

module Foreign.CUDA.Driver.Module.Base (

  -- * Module Management
  Module(..),
  JITOption(..), JITTarget(..), JITResult(..), JITFallback(..), JITInputType(..),
  JITOptionInternal(..),

  -- ** Loading and unloading modules
  loadFile,
  loadData,   loadDataFromPtr,
  loadDataEx, loadDataFromPtrEx,
  unload,

  -- Internal
  jitOptionUnpack, jitTargetOfCompute,

) where

#include "cbits/stubs.h"
{# context lib="cuda" #}

-- Friends
import Foreign.CUDA.Analysis.Device
import Foreign.CUDA.Driver.Error
import Foreign.CUDA.Internal.C2HS

-- System
import Foreign
import Foreign.C
import Unsafe.Coerce

import Control.Monad                                    ( liftM )
import Data.ByteString.Char8                            ( ByteString )
import qualified Data.ByteString.Char8                  as B
import qualified Data.ByteString.Internal               as B


--------------------------------------------------------------------------------
-- Data Types
--------------------------------------------------------------------------------

-- |
-- A reference to a Module object, containing collections of device functions
--
newtype Module = Module { useModule :: {# type CUmodule #}}
  deriving (Eq, Show)

-- |
-- Just-in-time compilation and linking options
--
data JITOption
  = MaxRegisters       !Int             -- ^ maximum number of registers per thread
  | ThreadsPerBlock    !Int             -- ^ number of threads per block to target for
  | OptimisationLevel  !Int             -- ^ level of optimisation to apply (1-4, default 4)
  | Target             !Compute         -- ^ compilation target, otherwise determined from context
  | FallbackStrategy   !JITFallback     -- ^ fallback strategy if matching cubin not found
  | GenerateDebugInfo                   -- ^ generate debug info (-g) (requires cuda >= 5.5)
  | GenerateLineInfo                    -- ^ generate line number information (-lineinfo) (requires cuda >= 5.5)
  | Verbose                             -- ^ verbose log messages (requires cuda >= 5.5)
  deriving (Show)

-- |
-- Results of online compilation
--
data JITResult = JITResult
  {
    jitTime     :: !Float,              -- ^ milliseconds spent compiling PTX
    jitInfoLog  :: !ByteString,         -- ^ information about PTX assembly
    jitModule   :: !Module              -- ^ the compiled module
  }
  deriving (Show)


-- |
-- Online compilation target architecture
--
{# enum CUjit_target as JITTarget
    { underscoreToCase }
    with prefix="CU_TARGET" deriving (Eq, Show) #}

-- |
-- Online compilation fallback strategy
--
{# enum CUjit_fallback as JITFallback
    { underscoreToCase
    , CU_PREFER_PTX as PreferPTX }
    with prefix="CU" deriving (Eq, Show) #}

-- |
-- Device code formats that can be used for online linking
--
#if CUDA_VERSION < 5050
data JITInputType
#else
{# enum CUjitInputType as JITInputType
    { underscoreToCase
    , CU_JIT_INPUT_PTX as PTX }
    with prefix="CU_JIT_INPUT" deriving (Eq, Show) #}
#endif

{# enum CUjit_option as JITOptionInternal
    { }
    with prefix="CU" deriving (Eq, Show) #}


--------------------------------------------------------------------------------
-- Module management
--------------------------------------------------------------------------------

-- |
-- Load the contents of the specified file (either a ptx or cubin file) to
-- create a new module, and load that module into the current context.
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html#group__CUDA__MODULE_1g366093bd269dafd0af21f1c7d18115d3>
--
{-# INLINEABLE loadFile #-}
loadFile :: FilePath -> IO Module
loadFile !ptx = resultIfOk =<< cuModuleLoad ptx

{-# INLINE cuModuleLoad #-}
{# fun unsafe cuModuleLoad
  { alloca-      `Module'   peekMod*
  , withCString* `FilePath'          } -> `Status' cToEnum #}


-- |
-- Load the contents of the given image into a new module, and load that module
-- into the current context. The image is (typically) the contents of a cubin or
-- PTX file.
--
-- Note that the 'ByteString' will be copied into a temporary staging area so
-- that it can be passed to C.
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html#group__CUDA__MODULE_1g04ce266ce03720f479eab76136b90c0b>
--
{-# INLINEABLE loadData #-}
loadData :: ByteString -> IO Module
loadData !img =
  B.useAsCString img (\p -> loadDataFromPtr (castPtr p))

-- |
-- As 'loadData', but read the image data from the given pointer. The image is a
-- NULL-terminated sequence of bytes.
--
{-# INLINEABLE loadDataFromPtr #-}
loadDataFromPtr :: Ptr Word8 -> IO Module
loadDataFromPtr !img = resultIfOk =<< cuModuleLoadData img

{-# INLINE cuModuleLoadData #-}
{# fun unsafe cuModuleLoadData
  { alloca- `Module'    peekMod*
  , castPtr `Ptr Word8'          } -> ` Status' cToEnum #}


-- |
-- Load the contents of the given image into a module with online compiler
-- options, and load the module into the current context. The image is
-- (typically) the contents of a cubin or PTX file. The actual attributes of the
-- compiled kernel can be probed using 'requires'.
--
-- Note that the 'ByteString' will be copied into a temporary staging area so
-- that it can be passed to C.
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html#group__CUDA__MODULE_1g9e8047e9dbf725f0cd7cafd18bfd4d12>
--
{-# INLINEABLE loadDataEx #-}
loadDataEx :: ByteString -> [JITOption] -> IO JITResult
loadDataEx !img !options =
  B.useAsCString img (\p -> loadDataFromPtrEx (castPtr p) options)

-- |
-- As 'loadDataEx', but read the image data from the given pointer. The image is
-- a NULL-terminated sequence of bytes.
--
{-# INLINEABLE loadDataFromPtrEx #-}
loadDataFromPtrEx :: Ptr Word8 -> [JITOption] -> IO JITResult
loadDataFromPtrEx !img !options = do
  let logSize = 2048

  fp_ilog <- B.mallocByteString logSize

  allocaArray logSize    $ \p_elog -> do
  withForeignPtr fp_ilog $ \p_ilog -> do

  let (opt,val) = unzip $
        [ (JIT_WALL_TIME, 0) -- must be first, this is extracted below
        , (JIT_INFO_LOG_BUFFER_SIZE_BYTES,  logSize)
        , (JIT_ERROR_LOG_BUFFER_SIZE_BYTES, logSize)
        , (JIT_INFO_LOG_BUFFER,  unsafeCoerce (p_ilog :: CString))
        , (JIT_ERROR_LOG_BUFFER, unsafeCoerce (p_elog :: CString))
        ]
        ++
        map jitOptionUnpack options

  withArrayLen (map cFromEnum opt)    $ \i p_opts -> do
  withArray    (map unsafeCoerce val) $ \  p_vals -> do

  (s,mdl) <- cuModuleLoadDataEx img i p_opts p_vals

  case s of
    Success -> do
      time    <- peek (castPtr p_vals)
      bytes   <- c_strnlen p_ilog logSize
      let infoLog | bytes == 0 = B.empty
                  | otherwise  = B.fromForeignPtr (castForeignPtr fp_ilog) 0 bytes
      return  $! JITResult time infoLog mdl

    _       -> do
      errLog  <- peekCString p_elog
      cudaError (unlines [describe s, errLog])


{-# INLINE cuModuleLoadDataEx #-}
{# fun unsafe cuModuleLoadDataEx
  { alloca- `Module'       peekMod*
  , castPtr `Ptr Word8'
  ,         `Int'
  , id      `Ptr CInt'
  , id      `Ptr (Ptr ())'          } -> `Status' cToEnum #}


-- |
-- Unload a module from the current context.
--
-- <http://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html#group__CUDA__MODULE_1g8ea3d716524369de3763104ced4ea57b>
--
{-# INLINEABLE unload #-}
unload :: Module -> IO ()
unload !m = nothingIfOk =<< cuModuleUnload m

{-# INLINE cuModuleUnload #-}
{# fun unsafe cuModuleUnload
  { useModule `Module' } -> `Status' cToEnum #}


--------------------------------------------------------------------------------
-- Internal
--------------------------------------------------------------------------------

{-# INLINE peekMod #-}
peekMod :: Ptr {# type CUmodule #} -> IO Module
peekMod = liftM Module . peek


{-# INLINE jitOptionUnpack #-}
jitOptionUnpack :: JITOption -> (JITOptionInternal, Int)
jitOptionUnpack (MaxRegisters x)      = (JIT_MAX_REGISTERS,       x)
jitOptionUnpack (ThreadsPerBlock x)   = (JIT_THREADS_PER_BLOCK,   x)
jitOptionUnpack (OptimisationLevel x) = (JIT_OPTIMIZATION_LEVEL,  x)
jitOptionUnpack (Target x)            = (JIT_TARGET,              fromEnum (jitTargetOfCompute x))
jitOptionUnpack (FallbackStrategy x)  = (JIT_FALLBACK_STRATEGY,   fromEnum x)
#if CUDA_VERSION >= 5050
jitOptionUnpack GenerateDebugInfo     = (JIT_GENERATE_DEBUG_INFO, fromEnum True)
jitOptionUnpack GenerateLineInfo      = (JIT_GENERATE_LINE_INFO,  fromEnum True)
jitOptionUnpack Verbose               = (JIT_LOG_VERBOSE,         fromEnum True)
#else
jitOptionUnpack GenerateDebugInfo     = requireSDK 'GenerateDebugInfo 5.5
jitOptionUnpack GenerateLineInfo      = requireSDK 'GenerateLineInfo 5.5
jitOptionUnpack Verbose               = requireSDK 'Verbose 5.5
#endif


{-# INLINE jitTargetOfCompute #-}
jitTargetOfCompute :: Compute -> JITTarget
jitTargetOfCompute (Compute x y) = toEnum (10*x + y)


#if defined(WIN32)
{-# INLINE c_strnlen' #-}
c_strnlen' :: CString -> CSize -> IO CSize
c_strnlen' str size = do
  str' <- peekCStringLen (str, fromIntegral size)
  return $ stringLen 0 str'
  where
    stringLen acc []       = acc
    stringLen acc ('\0':_) = acc
    stringLen acc (_:xs)   = stringLen (acc+1) xs
#else
foreign import ccall unsafe "string.h strnlen" c_strnlen'
  :: CString -> CSize -> IO CSize
#endif

{-# INLINE c_strnlen #-}
c_strnlen :: CString -> Int -> IO Int
c_strnlen str maxlen = cIntConv `fmap` c_strnlen' str (cIntConv maxlen)

