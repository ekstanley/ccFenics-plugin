||| Properties.idr -- Impossibility proofs and structural properties.
|||
||| Proves that certain invalid states cannot be constructed,
||| and that operations preserve structural properties.
|||
||| Key difference from Lean 4 (Layer 3):
|||   Lean 4: "given valid(s), prove valid(op(s))"
|||   Idris 2: "the TYPE of SessionState guarantees valid(s) always"
|||
||| These properties are corollaries of the type structure, not separate
||| theorems requiring explicit proof.
module DolfinxMCP.Properties

import DolfinxMCP.Registry
import DolfinxMCP.SessionState
import DolfinxMCP.Operations

%default total

-- ---------------------------------------------------------------------------
-- Property 1: freshState is valid
-- ---------------------------------------------------------------------------

||| freshState satisfies all invariants trivially.
||| (This is automatic from the type -- freshState has type SessionState.)
export
freshStateValid : SessionState
freshStateValid = freshState

-- ---------------------------------------------------------------------------
-- Property 2: cleanup always produces freshState
-- ---------------------------------------------------------------------------

||| cleanup on any state produces freshState.
||| Corresponds to P3 in test_property_invariants.py.
export
cleanupIsFresh : (s : SessionState) -> cleanup s = freshState
cleanupIsFresh _ = Refl

-- ---------------------------------------------------------------------------
-- Property 3: registerMesh preserves state validity
-- ---------------------------------------------------------------------------

||| registerMesh on any valid state produces a valid state.
||| (Automatic from the return type of registerMesh : SessionState -> SessionState.)
export
registerMeshValid : (name : String) -> (s : SessionState) -> SessionState
registerMeshValid = registerMesh

-- ---------------------------------------------------------------------------
-- Property 4: Cannot construct a space referencing a non-existent mesh
-- ---------------------------------------------------------------------------

||| To call registerFunctionSpace, you MUST provide HasKey proof.
||| Without that proof, the code does not compile.
||| This is INV-2 by construction -- no explicit proof needed.
|||
||| The "proof" is the type signature itself:
|||   registerFunctionSpace : ... -> HasKey meshName s.meshKeys -> ...
|||
||| Attempting to call registerFunctionSpace without a valid HasKey is
||| a TYPE ERROR, not a runtime error.

-- ---------------------------------------------------------------------------
-- Property 5: removeKey postcondition (from Registry.idr)
-- ---------------------------------------------------------------------------

||| After removeKey, the removed key is absent.
||| This is the type-level equivalent of the Python postcondition:
|||   assert mesh_name not in session.meshes
|||
||| Re-exported from Registry for visibility.
export
removedKeyAbsent : (k : String) -> (ks : List String)
                -> Not (HasKey k (removeKey k ks))
removedKeyAbsent = removeKeyAbsent

-- ---------------------------------------------------------------------------
-- Property 6: Unrelated keys survive removal
-- ---------------------------------------------------------------------------

||| If j /= k, then removing k from ks preserves j's membership.
||| Type-level equivalent of: removeMesh(m1) preserves m2.
export
unrelatedKeySurvives : {j, k : String} -> {ks : List String}
                    -> Not (j = k)
                    -> HasKey j ks
                    -> HasKey j (removeKey k ks)
unrelatedKeySurvives = removeKeyPreserves

-- ---------------------------------------------------------------------------
-- Property 7: Construction prevention (impossibility)
-- ---------------------------------------------------------------------------

||| It is impossible to construct a ValidRef for a key not in the list.
||| This is the fundamental guarantee: dangling references are type errors.
export
noDanglingRef : Not (HasKey k [])
noDanglingRef = \case _ impossible

||| Stronger: if k is not in ks, no ValidRef whose name equals k can exist.
||| This precisely captures the impossibility of a dangling reference.
export
noRefForAbsentKey : {k : String} -> {ks : List String}
                 -> Not (HasKey k ks)
                 -> (ref : ValidRef ks) -> Not (ref.name = k)
noRefForAbsentKey notHas (MkValidRef k prf) Refl = notHas prf
