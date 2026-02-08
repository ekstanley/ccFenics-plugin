import DolfinxProofs.State

/-!
# SessionState Mutation Operations

Pure-functional models of each Python mutation method.
Each function takes a SessionState and returns a new SessionState.
-/

namespace DolfinxProofs

/-- Empty session state. Corresponds to `SessionState.__init__()`. -/
def freshState : SessionState where
  meshes          := []
  function_spaces := []
  functions       := []
  bcs             := []
  solutions       := []
  mesh_tags       := []
  entity_maps     := []
  forms           := []
  ufl_symbols     := []
  active_mesh     := none

/-- Register a mesh. Sets active_mesh to the new name. -/
def registerMesh (s : SessionState) (name : String) : SessionState :=
  { s with
    meshes := name :: s.meshes
    active_mesh := some name }

/-- Register a function space. Precondition: meshName in meshes. -/
def registerFunctionSpace (s : SessionState) (name : String) (meshName : String) : SessionState :=
  { s with function_spaces := (name, meshName) :: s.function_spaces }

/-- Register a function. Precondition: spaceName is a key in function_spaces. -/
def registerFunction (s : SessionState) (name : String) (spaceName : String) : SessionState :=
  { s with functions := (name, spaceName) :: s.functions }

/-- Register a BC. Precondition: spaceName is a key in function_spaces. -/
def registerBC (s : SessionState) (name : String) (spaceName : String) : SessionState :=
  { s with bcs := (name, spaceName) :: s.bcs }

/-- Register a solution. Precondition: spaceName is a key in function_spaces. -/
def registerSolution (s : SessionState) (name : String) (spaceName : String) : SessionState :=
  { s with solutions := (name, spaceName) :: s.solutions }

/-- Register mesh tags. Precondition: meshName in meshes. -/
def registerMeshTags (s : SessionState) (name : String) (meshName : String) : SessionState :=
  { s with mesh_tags := (name, meshName) :: s.mesh_tags }

/-- Register an entity map. Precondition: parentMesh and childMesh in meshes. -/
def registerEntityMap (s : SessionState) (name : String) (parentMesh : String)
    (childMesh : String) : SessionState :=
  { s with entity_maps := (name, parentMesh, childMesh) :: s.entity_maps }

/-- Remove functions, BCs, and solutions that depend on a space. -/
def removeSpaceDeps (s : SessionState) (spaceName : String) : SessionState :=
  { s with
    functions := s.functions.filter (fun (_, sn) => !decide (sn = spaceName))
    bcs       := s.bcs.filter       (fun (_, sn) => !decide (sn = spaceName))
    solutions := s.solutions.filter  (fun (_, sn) => !decide (sn = spaceName)) }

/-- Remove a mesh with full cascade deletion. (session.py L514-573) -/
def removeMesh (s : SessionState) (name : String) : SessionState :=
  let depSpaceKeys :=
    (s.function_spaces.filter (fun (_, mn) => decide (mn = name))).map Prod.fst
  { meshes          := s.meshes.filter (fun m => !decide (m = name))
    function_spaces := s.function_spaces.filter (fun (_, mn) => !decide (mn = name))
    functions       := s.functions.filter  (fun (_, sn) => !depSpaceKeys.contains sn)
    bcs             := s.bcs.filter        (fun (_, sn) => !depSpaceKeys.contains sn)
    solutions       := s.solutions.filter  (fun (_, sn) => !depSpaceKeys.contains sn)
    mesh_tags       := s.mesh_tags.filter  (fun (_, mn) => !decide (mn = name))
    entity_maps     := s.entity_maps.filter
                          (fun (_, pm, cm) => !decide (pm = name) && !decide (cm = name))
    forms           := if depSpaceKeys == [] then s.forms else []
    ufl_symbols     := if depSpaceKeys == [] then s.ufl_symbols else []
    active_mesh     := if s.active_mesh = some name then none else s.active_mesh }

/-- Clear all state. -/
def cleanup (_ : SessionState) : SessionState := freshState

end DolfinxProofs
