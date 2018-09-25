{-# LANGUAGE BangPatterns             #-}
{-# LANGUAGE CPP                      #-}
{-# LANGUAGE EmptyDataDecls           #-}
{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE TemplateHaskell          #-}
--------------------------------------------------------------------------------
-- |
-- Module    : Foreign.CUDA.Driver.Graph.Build
-- Copyright : [2018] Trevor L. McDonell
-- License   : BSD
--
-- Graph construction functions for the low-level driver interface
--
-- Requires CUDA-10
--
--------------------------------------------------------------------------------

module Foreign.CUDA.Driver.Graph.Build (

  Graph(..), Node(..), NodeType(..), HostCallback,
  create, destroy, clone, remove,

  -- ** Construction
  addChild,
  addEmpty,
  addHost,
  addKernel,
  addMemcpy,
  addMemset,
  addDependencies,
  removeDependencies,

  -- ** Querying
  getType,
  getChildGraph,
  getEdges,
  getNodes,
  getRootNodes,
  getDependencies,
  getDependents,
  findInClone,

) where

#include "cbits/stubs.h"
{# context lib="cuda" #}

import Foreign.CUDA.Driver.Context.Base                   ( Context(..) )
import Foreign.CUDA.Driver.Error
import Foreign.CUDA.Driver.Exec                           ( Fun(..), FunParam(..) )
import Foreign.CUDA.Driver.Graph.Base
import Foreign.CUDA.Driver.Marshal                        ( useDeviceHandle )
import Foreign.CUDA.Driver.Stream                         ( Stream(..) )
import Foreign.CUDA.Driver.Unified                        ( MemoryType(..) )
import Foreign.CUDA.Internal.C2HS
import Foreign.CUDA.Ptr                                   ( DevicePtr(..) )

import Control.Monad                                      ( liftM )
import Unsafe.Coerce

import Data.Word
import Foreign
import Foreign.C
import Foreign.Storable


-- | Callback function executed on the host
--
-- <https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TYPES.html#group__CUDA__TYPES_1g262cd3570ff5d396db4e3dabede3c355>
--
-- @since 0.10.0.0
--
#if CUDA_VERSION < 10000
data HostCallback
#else
type HostCallback = {# type CUhostFn #}
#endif


--------------------------------------------------------------------------------
-- Graph creation
--------------------------------------------------------------------------------

-- | Create an empty task graph
--
-- Requires CUDA-10.0
--
-- <https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gd885f719186010727b75c3315f865fdf>
--
-- @since 0.10.0.0
--
{-# INLINEABLE create #-}
#if CUDA_VERSION < 10000
create :: [GraphFlag] -> IO Graph
create = requireSDK 'create 10.0
#else
{# fun unsafe cuGraphCreate as create
  { alloca-         `Graph'       peekGraph*
  , combineBitMasks `[GraphFlag]'
  }
  -> `()' checkStatus*- #}
#endif


-- | Destroy a graph, as well as all of its nodes
--
-- Requires CUDA-10.0
--
-- <https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g718cfd9681f078693d4be2426fd689c8>
--
-- @since 0.10.0.0
{-# INLINEABLE destroy #-}
#if CUDA_VERSION < 10000
destroy :: Graph -> IO ()
destroy = requireSDK 'destroy 10.0
#else
{# fun unsafe cuGraphDestroy as destroy
  { useGraph `Graph'
  }
  -> `()' checkStatus*- #}
#endif


-- | Clone a graph
--
-- Requires CUDA-10.0
--
-- <https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g3603974654e463f2231c71d9b9d1517e>
--
-- @since 0.10.0.0
--
{-# INLINEABLE clone #-}
#if CUDA_VERSION < 10000
clone :: Graph -> IO Graph
clone = requireSDK 'clone 10.0
#else
{# fun unsafe cuGraphClone as clone
  { alloca-  `Graph' peekGraph*
  , useGraph `Graph'
  }
  -> `()' checkStatus*- #}
#endif


-- | Remove a node from the graph
--
-- Requires CUDA-10.0
--
-- <https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g00ed16434d983d8f0011683eacaf19b9>
--
-- @since 0.10.0.0
--
{-# INLINEABLE remove #-}
#if CUDA_VERSION < 10000
remove :: Node -> IO ()
remove = requireSDK 'remove 10.0
#else
{# fun unsafe cuGraphDestroyNode as remove
  { useNode `Node'
  }
  -> `()' checkStatus*- #}
#endif


-- | Create a child graph node and add it to the graph
--
-- Requires CUDA-10.0
--
-- <https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g3f27c2e56e3d568b09f00d438e61ceb1>
--
-- @since 0.10.0.0
--
{-# INLINEABLE addChild #-}
addChild :: Graph -> Graph -> [Node] -> IO Node
#if CUDA_VERSION < 10000
addChild = requireSDK 'addChild 10.0
#else
addChild parent child dependencies = cuGraphAddChildGraphNode parent dependencies child
  where
    {# fun unsafe cuGraphAddChildGraphNode
      { alloca-           `Node' peekNode*
      , useGraph          `Graph'
      , withNodeArrayLen* `[Node]'&
      , useGraph          `Graph'
      }
      -> `()' checkStatus*- #}
#endif


-- | Add dependency edges to the graph
--
-- Requires CUDA-10.0
--
-- <https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g81bf1a6965f881be6ad8d21cfe0ee44f>
--
-- @since 0.10.0.0
--
{-# INLINEABLE addDependencies #-}
addDependencies :: Graph -> [(Node,Node)] -> IO ()
#if CUDA_VERSION < 10000
addDependencies = requireSDK 'addDependencies 10.0
#else
addDependencies !g !deps = cuGraphAddDependencies g from to
  where
    (from, to) = unzip deps

    {# fun unsafe cuGraphAddDependencies
      { useGraph          `Graph'
      , withNodeArray*    `[Node]'
      , withNodeArrayLen* `[Node]'&
      }
      -> `()' checkStatus*- #}
#endif


-- | Remove dependency edges from the graph
--
-- Requires CUDA-10.0
--
-- <https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g8ab696a6b3ccd99db47feba7e97fb579>
--
-- @since 0.10.0.0
--
{-# INLINE removeDependencies #-}
removeDependencies :: Graph -> [(Node,Node)] -> IO ()
#if CUDA_VERSION < 10000
removeDependencies = requireSDK 'removeDependencies 10.0
#else
removeDependencies !g !deps = cuGraphRemoveDependencies g from to
  where
    (from, to) = unzip deps

    {# fun unsafe cuGraphRemoveDependencies
      { useGraph          `Graph'
      , withNodeArray*    `[Node]'
      , withNodeArrayLen* `[Node]'&
      }
      -> `()' checkStatus*- #}
#endif


-- | Create an empty node and add it to the graph.
--
-- Requires CUDA-10.0
--
-- <https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g8a8681dbe97dbbb236ea5ebf3abe2ada>
--
-- @since 0.10.0.0
--
{-# INLINEABLE addEmpty #-}
#if CUDA_VERSION < 10000
addEmpty :: Graph -> [Node] -> IO Node
addEmpty = requireSDK 'addEmpty 10.0
#else
{# fun unsafe cuGraphAddEmptyNode as addEmpty
  { alloca-           `Node' peekNode*
  , useGraph          `Graph'
  , withNodeArrayLen* `[Node]'&
  }
  -> `()' checkStatus*- #}
#endif


-- | Creates a host execution node and adds it to the graph
--
-- Requires CUDA-10.0
--
-- <https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g1ba15c2fe1afb8897091ecec4202b597>
--
-- @since 0.10.0.0
--
{-# INLINEABLE addHost #-}
#if CUDA_VERSION < 10000
addHost :: Graph -> [Node] -> HostCallback -> Ptr () -> IO Node
addHost = requireSDK 'addHost 10.0
#else
{# fun unsafe cuGraphAddHostNode_simple as addHost
  { alloca-           `Node'         peekNode*
  , useGraph          `Graph'
  , withNodeArrayLen* `[Node]'&
  , id                `HostCallback'
  ,                   `Ptr ()'
  }
  -> `()' checkStatus*- #}
#endif


-- | Create a kernel execution node and adds it to the graph
--
-- Requires CUDA-10.0
--
-- <https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g886a9096293238937f2f3bc7f2d57635>
--
-- @since 0.10.0.0
--
{-# INLINEABLE addKernel #-}
addKernel
    :: Graph
    -> [Node]
    -> Fun
    -> (Int, Int, Int)  -- ^ grid dimension
    -> (Int, Int, Int)  -- ^ thread block dimensions
    -> Int              -- ^ shared memory (bytes)
    -> [FunParam]
    -> IO Node
#if CUDA_VERSION < 10000
addKernel = requireSDK 'addKernel 10.0
#else
addKernel !g !ns !fun (!gx,!gy,!gz) (!tx,!ty,!tz) !sm !args
  = withMany withFP args
  $ \pa -> withArray pa
  $ \pp -> cuGraphAddKernelNode_simple g ns fun gx gy gz tx ty tz sm pp
  where
    withFP :: FunParam -> (Ptr () -> IO b) -> IO b
    withFP !p !f = case p of
      IArg v -> with' v (f . castPtr)
      FArg v -> with' v (f . castPtr)
      VArg v -> with' v (f . castPtr)

    -- can't use the standard 'with' because 'alloca' will pass an undefined
    -- dummy argument when determining 'sizeOf' and 'alignment', but sometimes
    -- instances in Accelerate need to evaluate this argument.
    --
    with' :: Storable a => a -> (Ptr a -> IO b) -> IO b
    with' !val !f =
      allocaBytes (sizeOf val) $ \ptr -> do
        poke ptr val
        f ptr

    {# fun unsafe cuGraphAddKernelNode_simple
      { alloca-           `Node'    peekNode*
      , useGraph          `Graph'
      , withNodeArrayLen* `[Node]'&
      , useFun            `Fun'
      ,                   `Int'
      ,                   `Int'
      ,                   `Int'
      ,                   `Int'
      ,                   `Int'
      ,                   `Int'
      ,                   `Int'
      , id                `Ptr (Ptr ())'
      }
      -> `()' checkStatus*- #}
#endif


-- | Create a memcpy node and add it to the graph
--
-- Requires CUDA-10.0
--
-- <https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gdd521e1437c1c3ea8822f66a32ff1f94>
--
-- @since 0.10.0.0
--
#if CUDA_VERSION < 10000
addMemcpy :: Graph -> [Node] -> Context -> Int -> Int -> Int -> Int -> MemoryType -> Ptr a -> Int -> Int -> Int -> Int -> Int -> Int -> MemoryType -> Ptr a -> Int -> Int -> Int -> Int -> Int -> IO Node
addMemcpy = requireSDK 'addMemcpy 10.0
#else
{# fun unsafe cuGraphAddMemcpyNode_simple as addMemcpy
  { alloca-           `Node'    peekNode*
  , useGraph          `Graph'
  , withNodeArrayLen* `[Node]'&
  , useContext        `Context'
  ,                   `Int'         -- ^ srcXInBytes
  ,                   `Int'         -- ^ srcY
  ,                   `Int'         -- ^ srcZ
  ,                   `Int'         -- ^ srcLOD
  , cFromEnum         `MemoryType'  -- ^ source memory type
  , castPtr           `Ptr a'       -- ^ source ptr
  ,                   `Int'         -- ^ srcPitch
  ,                   `Int'         -- ^ srcHeight
  ,                   `Int'         -- ^ dstXInBytes
  ,                   `Int'         -- ^ dstY
  ,                   `Int'         -- ^ dstZ
  ,                   `Int'         -- ^ dstLOD
  , cFromEnum         `MemoryType'  -- ^ destination memory type
  , castPtr           `Ptr a'       -- ^ destination ptr
  ,                   `Int'         -- ^ dstPitch
  ,                   `Int'         -- ^ dstHeight
  ,                   `Int'         -- ^ widthInBytes
  ,                   `Int'         -- ^ height
  ,                   `Int'         -- ^ depth
  }
  -> `()' checkStatus*- #}
#endif


-- | Create a memset node and add it to the graph
--
-- Requires CUDA-10.0
--
-- <https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gac7f59961798f14a9f94f9f6b53cc3b7>
--
-- @since 0.10.0.0
--
{-# INLINEABLE addMemset #-}
addMemset
    :: Storable a
    => Graph
    -> [Node]
    -> Context
    -> DevicePtr a
    -> a
    -> Int      -- ^ height
    -> Int      -- ^ pitch
    -> Int      -- ^ width
    -> IO Node
#if CUDA_VERSION < 10000
addMemset = requireSDK 'addMemset 10.0
#else
addMemset !g !ns !ctx !dptr !val !h !p !w =
  cuGraphAddMemsetNode_simple g ns ctx dptr bytes h p val' w
  where
    bytes = sizeOf val

    val' :: Word32
    val' = case bytes of
             1 -> fromIntegral (unsafeCoerce val :: Word8)
             2 -> fromIntegral (unsafeCoerce val :: Word16)
             4 -> fromIntegral (unsafeCoerce val :: Word32)
             _ -> cudaError "can only memset 8-, 16-, and 32-bit values"

    {# fun unsafe cuGraphAddMemsetNode_simple
      { alloca-           `Node'    peekNode*
      , useGraph          `Graph'
      , withNodeArrayLen* `[Node]'&
      , useContext        `Context'
      , useDeviceHandle   `DevicePtr a'
      ,                   `Int'
      ,                   `Int'
      ,                   `Int'
      ,                   `Word32'
      ,                   `Int'
      }
      -> `()' checkStatus*- #}
#endif


--------------------------------------------------------------------------------
-- Query
--------------------------------------------------------------------------------

-- | Return the type of a node
--
-- Requires CUDA-10.0
--
-- <https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gdb1776d97aa1c9d5144774b29e4b8c3e>
--
-- @since 0.10.0.0
--
{-# INLINEABLE getType #-}
#if CUDA_VERSION < 10000
getType :: Node -> IO NodeType
getType = requireSDK 'getType 10.0
#else
{# fun unsafe cuGraphNodeGetType as getType
  { useNode `Node'
  , alloca- `NodeType' peekEnum*
  }
  -> `()' checkStatus*- #}
#endif


-- | Retrieve the embedded graph of a child sub-graph node
--
-- Requires CUDA-10.0
--
-- <https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gbe9fc9267316b3778ef0db507917b4fd>
--
-- @since 0.10.0.0
--
{-# INLINEABLE getChildGraph #-}
#if CUDA_VERSION < 10000
getChildGraph :: Node -> IO Graph
getChildGraph = requireSDK 'getChildGraph 10.0
#else
{# fun unsafe cuGraphChildGraphNodeGetGraph as getChildGraph
  { useNode `Node'
  , alloca- `Graph' peekGraph*
  }
  -> `()' checkStatus*- #}
#endif


-- | Return a graph's dependency edges
--
-- Requires CUDA-10.0
--
-- <https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g2b7bd71b0b2b8521f141996e0975a0d7>
--
-- @since 0.10.0.0
--
{-# INLINEABLE getEdges #-}
getEdges :: Graph -> IO [(Node, Node)]
#if CUDA_VERSION < 10000
getEdges = requireSDK 'getEdges 10.0
#else
getEdges !g =
  alloca $ \p_count -> do
    cuGraphGetEdges g nullPtr nullPtr p_count
    count <- peekIntConv p_count
    allocaArray  count $ \p_from -> do
     allocaArray count $ \p_to   -> do
       cuGraphGetEdges g p_from p_to p_count
       from <- peekArray count p_from
       to   <- peekArray count p_to
       return $ zip from to
  where
    {# fun unsafe cuGraphGetEdges
      { useGraph     `Graph'
      , castPtr      `Ptr Node'
      , castPtr      `Ptr Node'
      , id           `Ptr CULong'
      }
      -> `()' checkStatus*- #}
#endif


-- | Return a graph's nodes
--
-- Requires CUDA-10.0
--
-- <https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gfa35a8e2d2fc32f48dbd67ba27cf27e5>
--
-- @since 0.10.0.0
--
{-# INLINEABLE getNodes #-}
getNodes :: Graph -> IO [Node]
#if CUDA_VERSION < 10000
getNodes = requireSDK 'getNodes 10.0
#else
getNodes !g =
  alloca $ \p_count -> do
    cuGraphGetNodes g nullPtr p_count
    count <- peekIntConv p_count
    allocaArray count $ \p_nodes -> do
      cuGraphGetNodes g p_nodes p_count
      peekArray count p_nodes
  where
    {# fun unsafe cuGraphGetNodes
      { useGraph `Graph'
      , castPtr  `Ptr Node'
      , id       `Ptr CULong'
      }
      -> `()' checkStatus*- #}
#endif


-- | Returns the root nodes of a graph
--
-- Requires CUDA-10.0
--
-- <https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gf8517646bd8b39ab6359f8e7f0edffbd>
--
-- @since 0.10.0.0
--
{-# INLINEABLE getRootNodes #-}
getRootNodes :: Graph -> IO [Node]
#if CUDA_VERSION < 10000
getRootNodes = requireSDK 'getRootNodes 10.0
#else
getRootNodes g =
  alloca $ \p_count -> do
    cuGraphGetRootNodes g nullPtr p_count
    count <- peekIntConv p_count
    allocaArray count $ \p_nodes -> do
      cuGraphGetRootNodes g p_nodes p_count
      peekArray count p_nodes
  where
    {# fun unsafe cuGraphGetRootNodes
      { useGraph `Graph'
      , castPtr  `Ptr Node'
      , id       `Ptr CULong'
      }
      -> `()' checkStatus*- #}
#endif


-- | Return the dependencies of a node
--
-- Requires CUDA-10.0
--
-- <https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g048f4c0babcbba64a933fc277cd45083>
--
-- @since 0.10.0.0
--
{-# INLINEABLE getDependencies #-}
getDependencies :: Node -> IO [Node]
#if CUDA_VERSION < 10000
getDependencies = requireSDK 'getDependencies 10.0
#else
getDependencies !n =
  alloca $ \p_count -> do
    cuGraphNodeGetDependencies n nullPtr p_count
    count <- peekIntConv p_count
    allocaArray count $ \p_deps -> do
      cuGraphNodeGetDependencies n p_deps p_count
      peekArray count p_deps
  where
    {# fun unsafe cuGraphNodeGetDependencies
      { useNode `Node'
      , castPtr `Ptr Node'
      , id      `Ptr CULong'
      }
      -> `()' checkStatus*- #}
#endif


-- | Return a node's dependent nodes
--
-- Requires CUDA-10.0
--
-- <https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1g4b73d9e3b386a9c0b094a452b8431f59>
--
-- @since 0.10.0.0
--
{-# INLINEABLE getDependents #-}
getDependents :: Node -> IO [Node]
#if CUDA_VERSION < 10000
getDependents = requireSDK 'getDependents 10.0
#else
getDependents n =
  alloca $ \p_count -> do
    cuGraphNodeGetDependentNodes n nullPtr p_count
    count <- peekIntConv p_count
    allocaArray count $ \p_deps -> do
      cuGraphNodeGetDependentNodes n p_deps p_count
      peekArray count p_deps
  where
    {# fun unsafe cuGraphNodeGetDependentNodes
      { useNode `Node'
      , castPtr `Ptr Node'
      , id      `Ptr CULong'
      }
      -> `()' checkStatus*- #}
#endif


-- | Find a cloned version of a node
--
-- Requires CUDA-10.0
--
-- <https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html#group__CUDA__GRAPH_1gf21f6c968e346f028737c1118bfd41c2>
--
-- @since 0.10.0.0
--
{-# INLINEABLE findInClone #-}
#if CUDA_VERSION < 10000
findInClone :: Node -> Graph -> IO Node
findInClone = requireSDK 'findInClone 10
#else
{# fun unsafe cuGraphNodeFindInClone as findInClone
  { alloca-  `Node' peekNode*
  , useNode  `Node'
  , useGraph `Graph'
  }
  -> `()' checkStatus*- #}
#endif


-- TODO: since CUDA-10.0
--  * cuGraphHostNode[Get/Set]Params
--  * cuGraphKernelNode[Get/Set]Params
--  * cuGraphMemcpyNode[Get/Set]Params
--  * cuGraphMemsetNode[Get/Set]Params

--------------------------------------------------------------------------------
-- Internal
--------------------------------------------------------------------------------

#if CUDA_VERSION >= 10000
{-# INLINE peekGraph #-}
peekGraph :: Ptr {# type CUgraph #} -> IO Graph
peekGraph = liftM Graph . peek

{-# INLINE peekNode #-}
peekNode :: Ptr {# type CUgraphNode #} -> IO Node
peekNode = liftM Node . peek

{-# INLINE withNodeArray #-}
withNodeArray :: [Node] -> (Ptr {# type CUgraphNode #} -> IO a) -> IO a
withNodeArray ns f = withArray ns (f . castPtr)

{-# INLINE withNodeArrayLen #-}
withNodeArrayLen :: [Node] -> ((Ptr {# type CUgraphNode #}, CULong) -> IO a) -> IO a
withNodeArrayLen ns f = withArrayLen ns $ \i p -> f (castPtr p, cIntConv i)
#endif

