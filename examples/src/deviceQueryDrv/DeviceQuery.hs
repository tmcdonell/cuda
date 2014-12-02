{-# LANGUAGE RecordWildCards #-}

module Main where

import Numeric
import Control.Monad
import Text.Printf
import Text.PrettyPrint

import Foreign.CUDA.Analysis                            as CUDA
import qualified Foreign.CUDA.Driver                    as CUDA


main :: IO ()
main = do
  version       <- CUDA.driverVersion
  printf "CUDA device query (Driver API, statically linked)\n"
  printf "CUDA driver version %d.%d\n" (version`div`1000) ((version`mod`100)`div`10)

  CUDA.initialise []
  numDevices <- CUDA.count

  if numDevices == 0
     then printf "There are no available devices that support CUDA\n"
     else printf "Detected %d CUDA capable device%s\n" numDevices (if numDevices > 1 then "s" else "")

  forM_ [0 .. numDevices-1] $ \n -> do
    deviceProp <- CUDA.props =<< CUDA.device n
    printf "\nDevice %d: %s\n" n (deviceName deviceProp)
    statDevice deviceProp


statDevice :: DeviceProperties -> IO ()
statDevice dev@DeviceProperties{..} =
  let
      DeviceResources{..} = deviceResources dev

      pad v             = take width $ v ++ repeat ' '
      width             = maximum $ map (length . fst) props
      table             = nest 2 $ vcat $ map (\(k,v) -> text (pad k) <+> v) props

      grid (x,y)        = int x <+> char 'x' <+> int y
      cube (x,y,z)      = int x <+> char 'x' <+> int y <+> char 'x' <+> int z

      bool True         = text "Yes"
      bool False        = text "No"

      props =
        [("CUDA capability:",                    text $ show computeCapability)
        ,("CUDA cores:",                         text $ printf "%d cores in %d multiprocessors (%d cores/MP)" (coresPerMP * multiProcessorCount) multiProcessorCount coresPerMP)
        ,("Global memory:",                      text $ showBytes totalGlobalMem)
        ,("Constant memory:",                    text $ showBytes totalConstMem)
        ,("Shared memory per block:",            text $ showBytes sharedMemPerBlock)
        ,("Registers per block:",                int regsPerBlock)
        ,("Warp size:",                          int warpSize)
        ,("Maximum threads per multiprocessor:", int maxThreadsPerMultiProcessor)
        ,("Maximum threads per block:",          int maxThreadsPerBlock)
        ,("Maximum grid dimensions:",            cube maxGridSize)
        ,("Maximum block dimensions:",           cube maxBlockSize)
        ,("GPU clock rate:",                     text . showFreq $ clockRate * 1000)
        ,("Memory clock rate:",                  text . showFreq $ memClockRate * 1000)
        ,("Memory bus width:",                   int memBusWidth <> text "-bit")
        ,("L2 cache size:",                      text $ showBytes cacheMemL2)
        ,("Maximum texture dimensions",          empty)
        ,("  1D:",                               int maxTextureDim1D)
        ,("  2D:",                               grid maxTextureDim2D)
        ,("  3D:",                               cube maxTextureDim3D)
        ,("Texture alignment:",                  text $ showBytes textureAlignment)
        ,("Maximum memory pitch:",               text $ showBytes memPitch)
        ,("Concurrent kernel execution:",        bool concurrentKernels)
        ,("Concurrent copy and execution:",      bool deviceOverlap <> text (printf ", with %d copy engine%s" asyncEngineCount (if asyncEngineCount > 1 then "s" else "")))
        ,("Runtime limit on kernel execution:",  bool kernelExecTimeoutEnabled)
        ,("Integrated GPU sharing host memory:", bool integrated)
        ,("Host page-locked memory mapping:",    bool canMapHostMemory)
        ,("ECC memory support:",                 bool eccEnabled)
        ,("Unified addressing (UVA):",           bool unifiedAddressing)
        ,("PCI bus/location:",                   int (busID pciInfo) <> char '/' <> int (deviceID pciInfo))
        ,("Compute mode:",                       text (show computeMode))
        ]
  in
  putStrLn . render
    $ hang table 4
    $ case computeMode of
        Default          -> text "Multiple host threads can use the device simultaneously"
        Exclusive        -> text "Only one host thread in a process is able to use this device"
        Prohibited       -> text "No host thread can use this device"
        ExclusiveProcess -> text "Multiple threads in one process can use the device simultaneously"


showFreq :: Integral i => i -> String
showFreq x = showFFloatSIBase Nothing 1000 (fromIntegral x :: Double) "Hz"

showBytes :: Integral i => i -> String
showBytes x = showFFloatSIBase (Just 0) 1024 (fromIntegral x :: Double) "B"

showFFloatSIBase :: RealFloat a => Maybe Int -> a -> a -> ShowS
showFFloatSIBase p b n
  = showString
  $ showFFloat p n' (' ':si_unit)
  where
    n'          = n / (b ^^ pow)
    pow         = (-4) `max` floor (logBase b n) `min` 4        :: Int
    si_unit     = case pow of
                       -4 -> "p"
                       -3 -> "n"
                       -2 -> "µ"
                       -1 -> "m"
                       0  -> ""
                       1  -> "k"
                       2  -> "M"
                       3  -> "G"
                       4  -> "T"
                       _  -> error "out of range!"

