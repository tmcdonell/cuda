--------------------------------------------------------------------------------
-- |
-- Module    : PrettyPrint
-- Copyright : (c) 2009 Trevor L. McDonell
-- License   : BSD
--
-- Simple layout and pretty printing
--
--------------------------------------------------------------------------------

module PrettyPrint where

import Data.List
import Text.PrettyPrint
import System.IO


--------------------------------------------------------------------------------
-- Printing
--------------------------------------------------------------------------------

printDoc :: Doc -> IO ()
printDoc =  putStrLn . flip (++) "\n" . render

--
-- stolen from $fptools/ghc/compiler/utils/Pretty.lhs
--
-- This code has a BSD-style license
--
printDocFull :: Mode -> Handle -> Doc -> IO ()
printDocFull m hdl doc = do
  fullRender m cols 1.5 put done doc
  hFlush hdl
  where
    put (Chr c) next  = hPutChar hdl c >> next
    put (Str s) next  = hPutStr  hdl s >> next
    put (PStr s) next = hPutStr  hdl s >> next

    done = hPutChar hdl '\n'
    cols = 80


--------------------------------------------------------------------------------
-- Layout
--------------------------------------------------------------------------------

--
-- Display the given grid of renderable data, given as either a list of rows or
-- columns, using the minimum size required for each column. An additional
-- parameter specifies extra space to be inserted between each column.
--
ppAsRows      :: Int -> [[Doc]] -> Doc
ppAsRows q    =  ppAsColumns q . transpose

ppAsColumns   :: Int -> [[Doc]] -> Doc
ppAsColumns q =  vcat . map hsep . transpose . map (\col -> pad (width col) col)
  where
    len   = length . render
    width = maximum . map len
    pad w = map (\x -> x <> (hcat $ replicate (w - (len x) + q) space))

