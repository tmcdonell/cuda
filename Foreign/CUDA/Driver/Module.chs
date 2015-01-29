{-# LANGUAGE BangPatterns             #-}
{-# LANGUAGE ForeignFunctionInterface #-}
--------------------------------------------------------------------------------
-- |
-- Module    : Foreign.CUDA.Driver.Module
-- Copyright : [2009..2014] Trevor L. McDonell
-- License   : BSD
--
-- Module management for low-level driver interface
--
--------------------------------------------------------------------------------

module Foreign.CUDA.Driver.Module (

  -- * Module Management
  Module, JITOption(..), JITTarget(..), JITResult(..), JITFallback(..),

  -- ** Querying module inhabitants
  getFun, getPtr, getTex,

  -- ** Loading and unloading modules
  loadFile,
  loadData,   loadDataFromPtr,
  loadDataEx, loadDataFromPtrEx,
  unload

) where

#include "cbits/stubs.h"
{# context lib="cuda" #}

-- Friends
import Foreign.CUDA.Analysis.Device
import Foreign.CUDA.Ptr
import Foreign.CUDA.Driver.Error
import Foreign.CUDA.Driver.Exec
import Foreign.CUDA.Driver.Marshal                      ( peekDeviceHandle )
import Foreign.CUDA.Driver.Texture
import Foreign.CUDA.Internal.C2HS

-- System
import Foreign
import Foreign.C
import Unsafe.Coerce

import Control.Monad                                    ( liftM )
import Control.Exception                                ( throwIO )
import Data.Maybe                                       ( mapMaybe )
import Data.ByteString.Char8                            ( ByteString )
import qualified Data.ByteString.Char8                  as B
import qualified Data.ByteString.Internal               as B

#if CUDA_VERSION < 5050
import Debug.Trace                                      ( trace )
#endif

--------------------------------------------------------------------------------
-- Data Types
--------------------------------------------------------------------------------

-- |
-- A reference to a Module object, containing collections of device functions
--
newtype Module = Module { useModule :: {# type CUmodule #}}
  deriving (Eq, Show)


-- |
-- Just-in-time compilation options
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
    jitModule   :: !Module              -- ^ compilation error log or compiled module
  }
  deriving (Show)


{# enum CUjit_option as JITOptionInternal
    { }
    with prefix="CU" deriving (Eq, Show) #}

{# enum CUjit_target as JITTarget
    { underscoreToCase }
    with prefix="CU_TARGET" deriving (Eq, Show) #}

{# enum CUjit_fallback as JITFallback
    { underscoreToCase
    , CU_PREFER_PTX as PTX }
    with prefix="CU_PREFER" deriving (Eq, Show) #}


--------------------------------------------------------------------------------
-- Module management
--------------------------------------------------------------------------------

-- |
-- Returns a function handle
--
{-# INLINEABLE getFun #-}
getFun :: Module -> String -> IO Fun
getFun !mdl !fn = resultIfFound "function" fn =<< cuModuleGetFunction mdl fn

{-# INLINE cuModuleGetFunction #-}
{# fun unsafe cuModuleGetFunction
  { alloca-      `Fun'    peekFun*
  , useModule    `Module'
  , withCString* `String'          } -> `Status' cToEnum #}
  where peekFun = liftM Fun . peek


-- |
-- Return a global pointer, and size of the global (in bytes)
--
{-# INLINEABLE getPtr #-}
getPtr :: Module -> String -> IO (DevicePtr a, Int)
getPtr !mdl !name = do
  (!status,!dptr,!bytes) <- cuModuleGetGlobal mdl name
  resultIfFound "global" name (status,(dptr,bytes))

{-# INLINE cuModuleGetGlobal #-}
{# fun unsafe cuModuleGetGlobal
  { alloca-      `DevicePtr a' peekDeviceHandle*
  , alloca-      `Int'         peekIntConv*
  , useModule    `Module'
  , withCString* `String'                        } -> `Status' cToEnum #}


-- |
-- Return a handle to a texture reference
--
{-# INLINEABLE getTex #-}
getTex :: Module -> String -> IO Texture
getTex !mdl !name = resultIfFound "texture" name =<< cuModuleGetTexRef mdl name

{-# INLINE cuModuleGetTexRef #-}
{# fun unsafe cuModuleGetTexRef
  { alloca-      `Texture' peekTex*
  , useModule    `Module'
  , withCString* `String'           } -> `Status' cToEnum #}


-- |
-- Load the contents of the specified file (either a ptx or cubin file) to
-- create a new module, and load that module into the current context.
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
  fp_ilog <- B.mallocByteString logSize

  allocaArray logSize    $ \p_elog -> do
  withForeignPtr fp_ilog $ \p_ilog -> do

  let (opt,val) = unzip $
        [ (JIT_WALL_TIME, 0) -- must be first, this is extracted below
        , (JIT_INFO_LOG_BUFFER_SIZE_BYTES,  logSize)
        , (JIT_ERROR_LOG_BUFFER_SIZE_BYTES, logSize)
        , (JIT_INFO_LOG_BUFFER,  unsafeCoerce (p_ilog :: CString))
        , (JIT_ERROR_LOG_BUFFER, unsafeCoerce (p_elog :: CString)) ] ++ mapMaybe unpack options

  withArray (map cFromEnum opt)    $ \p_opts -> do
  withArray (map unsafeCoerce val) $ \p_vals -> do

  (s,mdl) <- cuModuleLoadDataEx img (length opt) p_opts p_vals

  case s of
    Success     -> do
      time    <- peek (castPtr p_vals)
      infoLog <- B.fromForeignPtr (castForeignPtr fp_ilog) 0 `fmap` c_strnlen p_ilog logSize
      return  $! JITResult time infoLog mdl

    _           -> do
      errLog  <- peekCString p_elog
      cudaError (unlines [describe s, errLog])

  where
    logSize = 2048

    unpack (MaxRegisters x)      = Just (JIT_MAX_REGISTERS,       x)
    unpack (ThreadsPerBlock x)   = Just (JIT_THREADS_PER_BLOCK,   x)
    unpack (OptimisationLevel x) = Just (JIT_OPTIMIZATION_LEVEL,  x)
    unpack (Target x)            = Just (JIT_TARGET,              jitTargetOfCompute x)
    unpack (FallbackStrategy x)  = Just (JIT_FALLBACK_STRATEGY,   fromEnum x)
#if CUDA_VERSION >= 5050
    unpack GenerateDebugInfo     = Just (JIT_GENERATE_DEBUG_INFO, fromEnum True)
    unpack GenerateLineInfo      = Just (JIT_GENERATE_LINE_INFO,  fromEnum True)
    unpack Verbose               = Just (JIT_LOG_VERBOSE,         fromEnum True)
#else
    unpack x                     = trace ("Warning: JITOption '" ++ show x ++ "' requires at least cuda-5.5") Nothing
#endif

    jitTargetOfCompute (Compute x y)
      = fromEnum
      $ case (x,y) of
          (1,0) -> Compute10
          (1,1) -> Compute11
          (1,2) -> Compute12
          (1,3) -> Compute13
          (2,0) -> Compute20
          (2,1) -> Compute21
          (3,0) -> Compute30
          (3,5) -> Compute35
          _     -> error ("Unknown JIT Target for Compute " ++ show (Compute x y))


{-# INLINE cuModuleLoadDataEx #-}
{# fun unsafe cuModuleLoadDataEx
  { alloca- `Module'       peekMod*
  , castPtr `Ptr Word8'
  ,         `Int'
  , id      `Ptr CInt'
  , id      `Ptr (Ptr ())'          } -> `Status' cToEnum #}


-- |
-- Unload a module from the current context
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

{-# INLINE resultIfFound #-}
resultIfFound :: String -> String -> (Status, a) -> IO a
resultIfFound kind name (!status,!result) =
  case status of
       Success  -> return result
       NotFound -> cudaError (kind ++ ' ' : describe status ++ ": " ++ name)
       _        -> throwIO (ExitCode status)

{-# INLINE peekMod #-}
peekMod :: Ptr {# type CUmodule #} -> IO Module
peekMod = liftM Module . peek

{-# INLINE c_strnlen' #-}
#if defined(WIN32)
c_strnlen' :: CString -> CSize -> IO CSize
c_strnlen' str size = do
  str' <- peekCStringLen (str, fromIntegral size)
  return $ stringLen 0 str'
  where stringLen acc []        = acc
        stringLen acc ('\0':xs) = acc
        stringLen acc (_:xs)    = stringLen (acc+1) xs
#else
foreign import ccall unsafe "string.h strnlen" c_strnlen'
  :: CString -> CSize -> IO CSize
#endif

{-# INLINE c_strnlen #-}
c_strnlen :: CString -> Int -> IO Int
c_strnlen str maxlen = cIntConv `fmap` c_strnlen' str (cIntConv maxlen)

