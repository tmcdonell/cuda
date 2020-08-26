{-# LANGUAGE BangPatterns               #-}
{-# LANGUAGE CPP                        #-}
{-# LANGUAGE EmptyDataDecls             #-}
{-# LANGUAGE ForeignFunctionInterface   #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE TemplateHaskell            #-}
#ifdef USE_EMPTY_CASE
{-# LANGUAGE EmptyCase                  #-}
#endif
--------------------------------------------------------------------------------
-- |
-- Module    : Foreign.CUDA.Driver.Graph.Base
-- Copyright : [2018..2020] Trevor L. McDonell
-- License   : BSD
--
-- Graph execution functions for the low-level driver interface
--
-- Requires CUDA-10
--
--------------------------------------------------------------------------------

module Foreign.CUDA.Driver.Graph.Base
  where

#include "cbits/stubs.h"
{# context lib="cuda" #}

import Foreign.Storable
import Foreign.Ptr


--------------------------------------------------------------------------------
-- Data Types
--------------------------------------------------------------------------------

#if CUDA_VERSION < 10000
data Graph
#else
newtype Graph = Graph { useGraph :: {# type CUgraph #}}
  deriving (Eq, Show)
#endif

data GraphFlag
instance Enum GraphFlag where
#ifdef USE_EMPTY_CASE
  toEnum   x = error ("GraphFlag.toEnum: Cannot match " ++ show x)
  fromEnum x = case x of {}
#endif

#if CUDA_VERSION < 10000
data Node
data NodeType

instance Enum NodeType where
#ifdef USE_EMPTY_CASE
  toEnum   x = error ("NodeType.toEnum: Cannot match " ++ show x)
  fromEnum x = case x of {}
#endif
#else
newtype Node = Node { useNode :: {# type CUgraphNode #}}
  deriving (Eq, Show, Storable)

{# enum CUgraphNodeType as NodeType
  { underscoreToCase
  , CU_GRAPH_NODE_TYPE_GRAPH as Subgraph
  }
  with prefix="CU_GRAPH_NODE_TYPE" deriving (Eq, Show, Bounded) #}
#endif


#if CUDA_VERSION < 10000
data Executable
#else
newtype Executable = Executable { useExecutable :: {# type CUgraphExec #}}
  deriving (Eq, Show)
#endif

