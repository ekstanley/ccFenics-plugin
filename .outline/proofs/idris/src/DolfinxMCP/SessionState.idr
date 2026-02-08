||| SessionState.idr -- Intrinsically valid session state.
|||
||| The type of SessionState encodes all 7 invariants.
||| Invalid states (dangling references) cannot be constructed -- they are
||| type errors at compile time.
|||
||| Corresponds to src/dolfinx_mcp/session.py:263-282.
module DolfinxMCP.SessionState

import DolfinxMCP.Registry

%default total

-- ---------------------------------------------------------------------------
-- SessionState: intrinsically valid by construction
-- ---------------------------------------------------------------------------

||| Central registry with dependent types encoding referential integrity.
|||
||| Each `refs` field is a list of ValidRefs proving that every foreign key
||| points to an existing entry. There is no need for a `check_invariants`
||| method -- the type system enforces invariants at construction time.
|||
||| Type parameters:
|||   meshKeys      -- list of registered mesh names
|||   spaceKeys     -- list of registered function space names
|||   funcKeys      -- list of registered function names
|||   bcKeys        -- list of registered BC names
|||   solKeys       -- list of registered solution names
|||   tagKeys       -- list of registered mesh tag names
|||   emapKeys      -- list of registered entity map names
public export
record SessionState where
  constructor MkSessionState

  -- Key lists (the "registry" of what exists)
  meshKeys  : List String
  spaceKeys : List String
  funcKeys  : List String
  bcKeys    : List String
  solKeys   : List String
  tagKeys   : List String
  emapKeys  : List String

  -- INV-1: active mesh is None or a valid mesh key
  activeMesh : Maybe (ValidRef meshKeys)

  -- INV-2: each function space references a valid mesh
  spaceRefs : List (ValidRef meshKeys)
  spaceRefsLen : length spaceRefs = length spaceKeys

  -- INV-3: each function references a valid function space
  funcRefs : List (ValidRef spaceKeys)
  funcRefsLen : length funcRefs = length funcKeys

  -- INV-4: each BC references a valid function space
  bcRefs : List (ValidRef spaceKeys)
  bcRefsLen : length bcRefs = length bcKeys

  -- INV-5: each solution references a valid function space
  solRefs : List (ValidRef spaceKeys)
  solRefsLen : length solRefs = length solKeys

  -- INV-6: each mesh tag references a valid mesh
  tagRefs : List (ValidRef meshKeys)
  tagRefsLen : length tagRefs = length tagKeys

  -- INV-7: each entity map references two valid meshes (parent, child)
  emapRefs : List (ValidRef meshKeys, ValidRef meshKeys)
  emapRefsLen : length emapRefs = length emapKeys

-- ---------------------------------------------------------------------------
-- Fresh state: trivially valid
-- ---------------------------------------------------------------------------

||| Empty session state. All registries empty, all invariants trivially satisfied.
||| Corresponds to SessionState.__init__().
public export
freshState : SessionState
freshState = MkSessionState
  [] [] [] [] [] [] []   -- key lists
  Nothing                -- active mesh
  [] Refl                -- space refs
  [] Refl                -- func refs
  [] Refl                -- bc refs
  [] Refl                -- sol refs
  [] Refl                -- tag refs
  [] Refl                -- emap refs
