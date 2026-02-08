||| Registry.idr -- Dependent registry types with compile-time key membership proofs.
|||
||| Core abstraction: `HasKey k ks` proves that string `k` appears in list `ks`.
||| `ValidRef keys` bundles a name with its membership proof.
|||
||| These types make dangling references a type error rather than a runtime bug.
module DolfinxMCP.Registry

import Decidable.Equality

%default total

-- ---------------------------------------------------------------------------
-- HasKey: proof that a key exists in a list
-- ---------------------------------------------------------------------------

||| Proof that key `k` is a member of list `ks`.
||| Corresponds to Python's `k in session.meshes` check.
public export
data HasKey : String -> List String -> Type where
  ||| Key is the head of the list
  Here  : HasKey k (k :: ks)
  ||| Key is somewhere in the tail
  There : HasKey k ks -> HasKey k (k' :: ks)

-- ---------------------------------------------------------------------------
-- ValidRef: a name bundled with proof of membership
-- ---------------------------------------------------------------------------

||| A reference to an entry that provably exists in the given key list.
||| Encodes INV-2 through INV-7: every foreign key must reference a valid entry.
public export
record ValidRef (keys : List String) where
  constructor MkValidRef
  name  : String
  proof : HasKey name keys

-- ---------------------------------------------------------------------------
-- Weakening: adding a key preserves existing membership proofs
-- ---------------------------------------------------------------------------

||| If `k` is in `ks`, then `k` is also in `(x :: ks)`.
||| Needed when registerMesh adds a new key -- all existing ValidRefs
||| must be lifted to the extended key list.
public export
weakenHasKey : HasKey k ks -> HasKey k (x :: ks)
weakenHasKey Here      = There Here
weakenHasKey (There p) = There (There p)

||| Weaken a ValidRef to accommodate an extended key list.
public export
weakenRef : ValidRef ks -> ValidRef (x :: ks)
weakenRef (MkValidRef n p) = MkValidRef n (weakenHasKey p)

-- ---------------------------------------------------------------------------
-- Decidable membership: runtime key lookup that produces proofs
-- ---------------------------------------------------------------------------

||| Decide whether `k` is in `ks`, returning a proof or refutation.
||| This bridges the runtime world (string lookup) with the type world (HasKey).
public export
decHasKey : (k : String) -> (ks : List String) -> Dec (HasKey k ks)
decHasKey k [] = No (\case _ impossible)
decHasKey k (x :: xs) with (decEq k x)
  decHasKey k (k :: xs) | Yes Refl = Yes Here
  decHasKey k (x :: xs) | No neq with (decHasKey k xs)
    decHasKey k (x :: xs) | No neq | Yes p  = Yes (There p)
    decHasKey k (x :: xs) | No neq | No np  = No (\case
      Here    => neq Refl
      There p => np p)

-- ---------------------------------------------------------------------------
-- Key removal: filtering a key from a list
-- ---------------------------------------------------------------------------

||| Remove all occurrences of `k` from `ks`.
||| Used by removeMesh to compute the new key list.
public export
removeKey : (k : String) -> (ks : List String) -> List String
removeKey k [] = []
removeKey k (x :: xs) with (decEq k x)
  removeKey k (k :: xs) | Yes Refl = removeKey k xs
  removeKey k (x :: xs) | No _     = x :: removeKey k xs

-- ---------------------------------------------------------------------------
-- Proof: removed key is absent from result
-- ---------------------------------------------------------------------------

||| After removeKey, the removed key does not appear in the result.
||| Corresponds to the postcondition of SessionState.remove_mesh:
||| "mesh_name not in session.meshes".
public export
removeKeyAbsent : (k : String) -> (ks : List String) -> Not (HasKey k (removeKey k ks))
removeKeyAbsent k [] = \case _ impossible
removeKeyAbsent k (x :: xs) with (decEq k x)
  removeKeyAbsent k (k :: xs) | Yes Refl = removeKeyAbsent k xs
  removeKeyAbsent k (x :: xs) | No neq = \case
    Here    => neq Refl
    There p => removeKeyAbsent k xs p

-- ---------------------------------------------------------------------------
-- Proof: other keys survive removal
-- ---------------------------------------------------------------------------

||| If `j /= k` and `j` is in `ks`, then `j` is in `removeKey k ks`.
||| Corresponds to: removeMesh preserves unrelated entries.
public export
removeKeyPreserves : {j, k : String} -> {ks : List String}
                  -> Not (j = k)
                  -> HasKey j ks
                  -> HasKey j (removeKey k ks)
removeKeyPreserves {ks = []} neq prf = absurd prf
removeKeyPreserves {ks = x :: xs} neq prf with (decEq k x)
  removeKeyPreserves {ks = k :: xs} neq Here         | Yes Refl = absurd (neq Refl)
  removeKeyPreserves {ks = k :: xs} neq (There rest)  | Yes Refl =
    removeKeyPreserves neq rest
  removeKeyPreserves {ks = x :: xs} neq Here         | No _ = Here
  removeKeyPreserves {ks = x :: xs} neq (There rest)  | No _ =
    There (removeKeyPreserves neq rest)
