||| Examples.idr -- Concrete worked examples demonstrating type safety.
|||
||| Each example constructs a SessionState through a sequence of operations,
||| showing that valid sequences type-check and invalid sequences do not compile.
module DolfinxMCP.Examples

import DolfinxMCP.Registry
import DolfinxMCP.SessionState
import DolfinxMCP.Operations

%default total

-- ---------------------------------------------------------------------------
-- Example 1: Register mesh, then function space referencing it
-- ---------------------------------------------------------------------------

||| Valid: register mesh "m1", then space "V1" on "m1".
||| The HasKey proof (Here) witnesses that "m1" is in the mesh list.
export
example1_validChain : SessionState
example1_validChain =
  let s0 = freshState
      s1 = registerMesh "m1" s0
      -- s1.meshKeys = ["m1"], so HasKey "m1" ["m1"] = Here
      s2 = registerFunctionSpace "V1" "m1" Here s1
  in s2

-- ---------------------------------------------------------------------------
-- Example 2: Full dependency chain
-- ---------------------------------------------------------------------------

||| Valid: mesh -> space -> function -> BC -> solution -> tags -> entity map.
||| Demonstrates all 7 invariants satisfied by construction.
export
example2_fullChain : SessionState
example2_fullChain =
  let s0 = freshState
      s1 = registerMesh "m1" s0
      s2 = registerMesh "m2" s1
      -- s2.meshKeys = ["m2", "m1"]
      -- HasKey "m1" ["m2", "m1"] = There Here
      -- HasKey "m2" ["m2", "m1"] = Here
      s3 = registerFunctionSpace "V1" "m1" (There Here) s2
      -- s3.spaceKeys = ["V1"]
      -- HasKey "V1" ["V1"] = Here
      s4 = registerFunction "f1" "V1" Here s3
      s5 = registerBC "bc1" "V1" Here s4
      s6 = registerSolution "u1" "V1" Here s5
      s7 = registerMeshTags "t1" "m1" (There Here) s6
      s8 = registerEntityMap "em1" "m2" "m1" Here (There Here) s7
  in s8

-- ---------------------------------------------------------------------------
-- Example 3: Cleanup produces fresh state
-- ---------------------------------------------------------------------------

||| cleanup on any state returns freshState.
export
example3_cleanup : SessionState
example3_cleanup =
  let s = example2_fullChain
  in cleanup s  -- Type: SessionState, value: freshState

-- ---------------------------------------------------------------------------
-- Example 4: Invalid construction (DOES NOT COMPILE)
-- ---------------------------------------------------------------------------

-- The following would be a type error if uncommented:
--
--   badExample : SessionState
--   badExample =
--     let s0 = freshState
--         -- s0.meshKeys = [], so no HasKey proof exists for any mesh name
--         s1 = registerFunctionSpace "V1" "m1" ??? s0
--         -- ??? : HasKey "m1" []  -- IMPOSSIBLE to construct!
--     in s1
--
-- This demonstrates construction prevention: you cannot create a
-- function space referencing a non-existent mesh. The type system
-- rejects it at compile time.

-- ---------------------------------------------------------------------------
-- Example 5: Two meshes, space on second
-- ---------------------------------------------------------------------------

||| Valid: register m1, m2, then space on m2 (head of meshKeys).
export
example5_spacesOnDifferentMeshes : SessionState
example5_spacesOnDifferentMeshes =
  let s0 = freshState
      s1 = registerMesh "m1" s0
      s2 = registerMesh "m2" s1
      -- s2.meshKeys = ["m2", "m1"]
      -- Space on m2: HasKey "m2" ["m2", "m1"] = Here
      s3 = registerFunctionSpace "V1" "m2" Here s2
      -- Space on m1: HasKey "m1" ["m2", "m1"] = There Here
      s4 = registerFunctionSpace "V2" "m1" (There Here) s3
  in s4
