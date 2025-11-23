--------------------------------------------------------------------------------
--
-- Module    : Fold
-- Copyright : (c) 2009 Trevor L. McDonell
-- License   : BSD
--
-- Reduce a vector to a single value
--
--------------------------------------------------------------------------------

module Main where

-- Friends
import PrettyPrint

-- System
import Numeric
import Data.List
import Control.Monad
import Control.Exception
import System.Exit
import System.Environment
import System.Console.GetOpt
import Text.PrettyPrint

import Foreign
import qualified Foreign.CUDA               as CUDA
import qualified Foreign.CUDA.Runtime.Event as CUDA


--------------------------------------------------------------------------------
-- Options
--------------------------------------------------------------------------------

data TestMode = Quick | Range | Shmoo
  deriving (Eq,Show,Read)

data MemoryMode = List | Pageable | Pinned | WriteCombined
  deriving (Eq,Show,Read,Ord,Enum)

data CopyMode = HostToDevice | DeviceToHost | DeviceToDevice
  deriving (Eq,Show,Read,Ord,Enum)

data Options = Options
  { testMode   :: TestMode
--  , memoryMode :: [MemoryMode]
--  , copyMode   :: [CopyMode]
  , device     :: Int
  , range      :: (Int,Int,Int)
  }
  deriving (Show)

defaultOptions :: Options
defaultOptions = Options
  { device     = 0
  , testMode   = Quick
--  , memoryMode = []
--  , copyMode   = []
  , range      = (kilobyte, kilobyte, 10*kilobyte)
  }

kilobyte, megabyte :: Int
kilobyte = 1 `shift` 10
megabyte = 1 `shift` 20

options :: [OptDescr (Options -> Options)]
options =
  [ Option ['d'] ["device"]    (ReqArg (\d opts -> opts { device = read d }) "ID")                           "numeral of device to test"
  , Option ['t'] ["test"]      (ReqArg (\t opts -> opts { testMode = read t}) "MODE")                        "testing mode: Quick | Range | Shmoo"
--  , Option ['m'] ["memory"]    (ReqArg (\m opts -> opts { memoryMode = read m : memoryMode opts}) "MODE")    "memory copy mode: List | Pageable | Pinned | WriteCombined"
--  , Option ['c'] ["copy"]      (ReqArg (\c opts -> opts { copyMode = read c : copyMode opts}) "DIRECTION")   "memory copy direction: DeviceToHost | HostToDevice | DeviceToDevice"
  , Option ['s'] ["start"]     (ReqArg (\s opts -> opts { range = let (_,i,e) = range opts in (read s,i,e)}) "BYTES") "starting transfer size"
  , Option ['i'] ["increment"] (ReqArg (\i opts -> opts { range = let (s,_,e) = range opts in (s,read i,e)}) "BYTES") "transfer test size increment"
  , Option ['e'] ["end"]       (ReqArg (\e opts -> opts { range = let (s,i,_) = range opts in (s,i,read e)}) "BYTES") "ending transfer size"
  ]


--------------------------------------------------------------------------------
-- Testing
--------------------------------------------------------------------------------

-- Benchmarking
--
bench :: Int -> IO a -> IO Double
bench n testee =
  let iter = 10
      size = fromIntegral (iter * n) * fromIntegral (sizeOf (undefined::Int))
  in
  bracket (CUDA.create []) CUDA.destroy $ \start ->
  bracket (CUDA.create []) CUDA.destroy $ \stop  -> do
    CUDA.record start Nothing
    replicateM_ iter testee
    CUDA.record stop  Nothing
    CUDA.sync
    ms <- realToFrac `fmap` CUDA.elapsedTime start stop
    return $ 1E3 * size / (fromIntegral megabyte * ms)


-- Bandwidth testing for the various copy modes
--
bandwidth :: CopyMode -> MemoryMode -> Int -> IO Double

bandwidth HostToDevice List     n =
  bench n (CUDA.withListArray [1..n] (\_ -> return ()))

bandwidth HostToDevice Pageable n =
  CUDA.allocaArray n $ \d_ptr ->
  withArray [1..n]   $ \h_ptr ->
  bench n (CUDA.pokeArray n h_ptr d_ptr)

bandwidth HostToDevice x n        =
  CUDA.allocaArray n $ \d_ptr ->
  let f = if x == WriteCombined then [CUDA.WriteCombined] else [] in
  bracket (CUDA.mallocHostArray f n) (CUDA.freeHost) $ \h_ptr -> do
  pokeArray (CUDA.useHostPtr h_ptr) [1..n]
  bench n (CUDA.pokeArrayAsync n h_ptr d_ptr Nothing)

bandwidth DeviceToHost List     n =
  CUDA.withListArray [1..n] $ \d_ptr ->
  bench n (CUDA.peekListArray n d_ptr)

bandwidth DeviceToHost Pageable n =
  allocaArray n             $ \h_ptr ->
  CUDA.withListArray [1..n] $ \d_ptr ->
  bench n (CUDA.peekArray n d_ptr h_ptr)

bandwidth DeviceToHost x n        =
  let f = if x == WriteCombined then [CUDA.WriteCombined] else [] in
  bracket (CUDA.mallocHostArray f n) (CUDA.freeHost) $ \h_ptr ->
  CUDA.withListArray [1..n]                          $ \d_ptr ->
  bench n (CUDA.peekArrayAsync n d_ptr h_ptr Nothing)

bandwidth DeviceToDevice _ n      =
  CUDA.withListArray [1..n] $ \d_src ->
  CUDA.allocaArray n        $ \d_dst ->
  (\x->2*x) `fmap` bench n (CUDA.copyArray n d_src d_dst)


-- Testing modes
--
runTests :: [Int] -> IO [(String,[Double])]
runTests bytes = sequence $
  [ run m c | m <- [List ..], c <- [HostToDevice,DeviceToHost]] ++ [ run Pageable DeviceToDevice ]
  where
    run m c =
      mapM (\b -> bandwidth c m (b `div` sizeOf (undefined::Int))) bytes >>= \t ->
      return (sc c ++ sm m, t)

    sc DeviceToHost   = "D->H"
    sc HostToDevice   = "H->D"
    sc DeviceToDevice = "D->D"
    sm List           = " (List)"
    sm Pinned         = " (Pinned)"
    sm WriteCombined  = " (WC)"
    sm Pageable       = ""


--------------------------------------------------------------------------------
-- Main
--------------------------------------------------------------------------------

printQuick :: [(String,[Double])] -> IO ()
printQuick = printDoc . ppAsRows 1 . (++) header . map k
  where
    k (x,y) = text x : map float' y
    float'  = text . flip (showFFloat (Just 2)) ""
    header  = map (map text) [["Mode", "Bandwidth (MB/s)"]
                             ,["----", "----------------"]]

printMany :: [Int] -> [(String,[Double])] -> IO ()
printMany xs tests =
  printDoc . ppAsRows 1 . (++) header . zipWith k xs . transpose . snd . unzip $ tests
  where
    k b ys  = int b : map float' ys
    float'  = text . flip (showFFloat (Just 2)) ""

    titles    = "Size (bytes)" : fst (unzip tests)
    seperator = map (flip replicate '-' . length) $ titles
    header    = map (map text) [titles,seperator]


parseOptions :: [String] -> IO (Options, [String])
parseOptions argv =
  case getOpt Permute options argv of
    (o,n,[])   -> return (foldl (flip id) defaultOptions o, n)
    (_,_,errs) -> do putStrLn $ concat errs ++ usageInfo header options
                     exitFailure
  where
    header = "Usage: bandwidthTest [OPTION...]"

shmooBytes :: [Int]
shmooBytes =
  [    kilobyte,   2*kilobyte  ..  20*kilobyte ] ++
  [ 22*kilobyte,  24*kilobyte  ..  50*kilobyte ] ++
  [ 60*kilobyte,  70*kilobyte  .. 100*kilobyte ] ++
  [ 200*kilobyte, 300*kilobyte ..   1*megabyte ] ++
  [   2*megabyte,   3*megabyte ..  16*megabyte ] ++
  [  18*megabyte,  20*megabyte ..  32*megabyte ] ++
  [  36*megabyte,  40*megabyte ..  64*megabyte ]

-- Main
--
main :: IO ()
main = do
  (opts,_) <- parseOptions =<< getArgs
  props    <- CUDA.props (device opts)
  putStrLn $  "Device " ++ show (device opts) ++ ": " ++ (CUDA.deviceName props)

  let (s,i,e) = range opts
      bytes   = [s,(s+i)..e]

  case testMode opts of
    Quick -> putStrLn "Quick mode: Bandwidth of 32MB transfer"
    _     -> putStrLn "Bandwidth measured in MB/s"

  putStrLn "Please wait...\n"
  case testMode opts of
    Range -> runTests bytes         >>= printMany bytes
    Shmoo -> runTests shmooBytes    >>= printMany shmooBytes
    Quick -> runTests [32*megabyte] >>= printQuick

