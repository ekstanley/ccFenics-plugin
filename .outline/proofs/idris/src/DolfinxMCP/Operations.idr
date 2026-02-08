||| Operations.idr -- Type-safe state mutations with proof obligations.
|||
||| Each operation takes a valid SessionState and returns a valid SessionState.
||| The type signatures guarantee that invariants are preserved -- no separate
||| proof step required (unlike Lean 4 Layer 3, which proves preservation
||| as external theorems).
|||
||| Register operations add keys and provide new ValidRefs.
||| The cleanup operation returns freshState.
module DolfinxMCP.Operations

import DolfinxMCP.Registry
import DolfinxMCP.SessionState

%default total

-- ---------------------------------------------------------------------------
-- registerMesh: add a mesh, set as active
-- ---------------------------------------------------------------------------

||| Register a new mesh. The mesh name is added to meshKeys, and
||| all existing ValidRef meshKeys are weakened to the extended list.
|||
||| Corresponds to: apply_operation(session, ("register_mesh", name))
public export
registerMesh : (name : String) -> SessionState -> SessionState
registerMesh name s = MkSessionState
  (name :: s.meshKeys)                                   -- meshKeys extended
  s.spaceKeys s.funcKeys s.bcKeys s.solKeys s.tagKeys s.emapKeys
  (Just (MkValidRef name Here))                          -- active = new mesh
  (map weakenRef s.spaceRefs) (weakenMapLen s.spaceRefs s.spaceKeys s.spaceRefsLen)
  s.funcRefs s.funcRefsLen
  s.bcRefs s.bcRefsLen
  s.solRefs s.solRefsLen
  (map weakenRef s.tagRefs) (weakenMapLen s.tagRefs s.tagKeys s.tagRefsLen)
  (map (\(p, c) => (weakenRef p, weakenRef c)) s.emapRefs)
    (weakenPairMapLen s.emapRefs s.emapKeys s.emapRefsLen)
  where
    weakenMapLen : (refs : List a) -> (keys : List b)
               -> length refs = length keys
               -> length (map f refs) = length keys
    weakenMapLen [] [] Refl = Refl
    weakenMapLen (_ :: rs) (_ :: ks) prf =
      cong S (weakenMapLen rs ks (succInjective _ _ prf))

    weakenPairMapLen : (refs : List (a, b)) -> (keys : List c)
                    -> length refs = length keys
                    -> length (map g refs) = length keys
    weakenPairMapLen [] [] Refl = Refl
    weakenPairMapLen (_ :: rs) (_ :: ks) prf =
      cong S (weakenPairMapLen rs ks (succInjective _ _ prf))

-- ---------------------------------------------------------------------------
-- registerFunctionSpace: requires proof that mesh exists
-- ---------------------------------------------------------------------------

||| Register a function space. Caller must provide a HasKey proof that
||| meshName exists in s.meshKeys. This is INV-2 by construction.
|||
||| Corresponds to: apply_operation(session, ("register_space", name, meshName))
||| In Python, the precondition is: meshName in session.meshes
||| In Idris 2, the precondition is: HasKey meshName s.meshKeys (compile-time)
public export
registerFunctionSpace : (name : String) -> (meshName : String)
                     -> HasKey meshName s.meshKeys
                     -> (s : SessionState) -> SessionState
registerFunctionSpace name meshName meshProof s = MkSessionState
  s.meshKeys
  (name :: s.spaceKeys)                                  -- spaceKeys extended
  s.funcKeys s.bcKeys s.solKeys s.tagKeys s.emapKeys
  s.activeMesh
  (MkValidRef meshName meshProof :: s.spaceRefs)         -- new ref with proof
    (cong S s.spaceRefsLen)
  (map weakenFuncRef s.funcRefs)                         -- weaken func refs
    (weakenMapLen s.funcRefs s.funcKeys s.funcRefsLen)
  (map weakenBCRef s.bcRefs)
    (weakenMapLen s.bcRefs s.bcKeys s.bcRefsLen)
  (map weakenSolRef s.solRefs)
    (weakenMapLen s.solRefs s.solKeys s.solRefsLen)
  s.tagRefs s.tagRefsLen
  s.emapRefs s.emapRefsLen
  where
    weakenFuncRef : ValidRef s.spaceKeys -> ValidRef (name :: s.spaceKeys)
    weakenFuncRef = weakenRef

    weakenBCRef : ValidRef s.spaceKeys -> ValidRef (name :: s.spaceKeys)
    weakenBCRef = weakenRef

    weakenSolRef : ValidRef s.spaceKeys -> ValidRef (name :: s.spaceKeys)
    weakenSolRef = weakenRef

    weakenMapLen : (refs : List a) -> (keys : List b)
               -> length refs = length keys
               -> length (map f refs) = length keys
    weakenMapLen [] [] Refl = Refl
    weakenMapLen (_ :: rs) (_ :: ks) prf =
      cong S (weakenMapLen rs ks (succInjective _ _ prf))

-- ---------------------------------------------------------------------------
-- registerFunction: requires proof that space exists
-- ---------------------------------------------------------------------------

||| Register a function. Caller must provide proof that spaceName exists
||| in s.spaceKeys. This is INV-3 by construction.
public export
registerFunction : (name : String) -> (spaceName : String)
                -> HasKey spaceName s.spaceKeys
                -> (s : SessionState) -> SessionState
registerFunction name spaceName spaceProof s = MkSessionState
  s.meshKeys s.spaceKeys
  (name :: s.funcKeys)                                   -- funcKeys extended
  s.bcKeys s.solKeys s.tagKeys s.emapKeys
  s.activeMesh
  s.spaceRefs s.spaceRefsLen
  (MkValidRef spaceName spaceProof :: s.funcRefs)        -- new ref with proof
    (cong S s.funcRefsLen)
  s.bcRefs s.bcRefsLen
  s.solRefs s.solRefsLen
  s.tagRefs s.tagRefsLen
  s.emapRefs s.emapRefsLen

-- ---------------------------------------------------------------------------
-- registerBC: requires proof that space exists
-- ---------------------------------------------------------------------------

||| Register a boundary condition. INV-4 by construction.
public export
registerBC : (name : String) -> (spaceName : String)
          -> HasKey spaceName s.spaceKeys
          -> (s : SessionState) -> SessionState
registerBC name spaceName spaceProof s = MkSessionState
  s.meshKeys s.spaceKeys s.funcKeys
  (name :: s.bcKeys)                                     -- bcKeys extended
  s.solKeys s.tagKeys s.emapKeys
  s.activeMesh
  s.spaceRefs s.spaceRefsLen
  s.funcRefs s.funcRefsLen
  (MkValidRef spaceName spaceProof :: s.bcRefs)          -- new ref with proof
    (cong S s.bcRefsLen)
  s.solRefs s.solRefsLen
  s.tagRefs s.tagRefsLen
  s.emapRefs s.emapRefsLen

-- ---------------------------------------------------------------------------
-- registerSolution: requires proof that space exists
-- ---------------------------------------------------------------------------

||| Register a solution. INV-5 by construction.
public export
registerSolution : (name : String) -> (spaceName : String)
                -> HasKey spaceName s.spaceKeys
                -> (s : SessionState) -> SessionState
registerSolution name spaceName spaceProof s = MkSessionState
  s.meshKeys s.spaceKeys s.funcKeys s.bcKeys
  (name :: s.solKeys)                                    -- solKeys extended
  s.tagKeys s.emapKeys
  s.activeMesh
  s.spaceRefs s.spaceRefsLen
  s.funcRefs s.funcRefsLen
  s.bcRefs s.bcRefsLen
  (MkValidRef spaceName spaceProof :: s.solRefs)         -- new ref with proof
    (cong S s.solRefsLen)
  s.tagRefs s.tagRefsLen
  s.emapRefs s.emapRefsLen

-- ---------------------------------------------------------------------------
-- registerMeshTags: requires proof that mesh exists
-- ---------------------------------------------------------------------------

||| Register mesh tags. INV-6 by construction.
public export
registerMeshTags : (name : String) -> (meshName : String)
                -> HasKey meshName s.meshKeys
                -> (s : SessionState) -> SessionState
registerMeshTags name meshName meshProof s = MkSessionState
  s.meshKeys s.spaceKeys s.funcKeys s.bcKeys s.solKeys
  (name :: s.tagKeys)                                    -- tagKeys extended
  s.emapKeys
  s.activeMesh
  s.spaceRefs s.spaceRefsLen
  s.funcRefs s.funcRefsLen
  s.bcRefs s.bcRefsLen
  s.solRefs s.solRefsLen
  (MkValidRef meshName meshProof :: s.tagRefs)           -- new ref with proof
    (cong S s.tagRefsLen)
  s.emapRefs s.emapRefsLen

-- ---------------------------------------------------------------------------
-- registerEntityMap: requires proofs that both meshes exist
-- ---------------------------------------------------------------------------

||| Register an entity map. INV-7 by construction.
public export
registerEntityMap : (name : String) -> (parent, child : String)
                 -> HasKey parent s.meshKeys
                 -> HasKey child s.meshKeys
                 -> (s : SessionState) -> SessionState
registerEntityMap name parent child pProof cProof s = MkSessionState
  s.meshKeys s.spaceKeys s.funcKeys s.bcKeys s.solKeys s.tagKeys
  (name :: s.emapKeys)                                   -- emapKeys extended
  s.activeMesh
  s.spaceRefs s.spaceRefsLen
  s.funcRefs s.funcRefsLen
  s.bcRefs s.bcRefsLen
  s.solRefs s.solRefsLen
  s.tagRefs s.tagRefsLen
  ((MkValidRef parent pProof, MkValidRef child cProof) :: s.emapRefs)
    (cong S s.emapRefsLen)

-- ---------------------------------------------------------------------------
-- cleanup: returns freshState (trivially valid)
-- ---------------------------------------------------------------------------

||| Reset to empty state. Corresponds to SessionState.cleanup().
||| Return type is freshState -- all registries empty, all invariants trivial.
public export
cleanup : SessionState -> SessionState
cleanup _ = freshState
