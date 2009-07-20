{-
 - Top level binding to CUDA library
 -}

module Foreign.CUDA
  (
--    withArrayD
  ) where

-- import Foreign
--
-- import qualified Foreign.CUDA.Runtime as C
-- import Foreign.CUDA.Types
-- import Foreign.CUDA.Utils
-- 
-- --------------------------------------------------------------------------------
-- -- Allocation and marshalling
-- --------------------------------------------------------------------------------
-- 
-- withArrayD       :: (Storable a) => [a] -> (DevicePtr -> IO b) -> IO b
-- withArrayD vec f =
--     withArrayLen vec $ \len hp ->
--       let bytes = toInteger $ len * sizeOf (head vec) in
--       forceEither `fmap` C.malloc bytes       >>= \dp  ->
--       newDevicePtr finalizerFree (castPtr hp) >>= \hp' ->
--       C.memcpy hp' dp bytes HostToDevice      >>
--       f dp >>= return
-- 
-- 
-- allocaArrayD         :: Int -> (DevicePtr -> IO b) -> IO b
-- allocaArrayD bytes f =
--     f =<< forceEither `fmap` C.malloc (toInteger bytes)

