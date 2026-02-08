/-!
# SessionState Model

Lean 4 model of the Python SessionState class from
`src/dolfinx_mcp/session.py`. Each registry stores only
the foreign key information needed for invariant proofs.

Correspondence to Python:
  meshes         : dict[str, MeshInfo]          --> List String (keys only)
  function_spaces: dict[str, FunctionSpaceInfo]  --> List (String x String) (key, mesh_name)
  functions      : dict[str, FunctionInfo]       --> List (String x String) (key, space_name)
  bcs            : dict[str, BCInfo]             --> List (String x String) (key, space_name)
  solutions      : dict[str, SolutionInfo]       --> List (String x String) (key, space_name)
  mesh_tags      : dict[str, MeshTagsInfo]       --> List (String x String) (key, mesh_name)
  entity_maps    : dict[str, EntityMapInfo]      --> List (String x String x String) (key, parent, child)
  active_mesh    : str | None                    --> Option String
-/

namespace DolfinxProofs

structure SessionState where
  meshes          : List String
  function_spaces : List (String × String)
  functions       : List (String × String)
  bcs             : List (String × String)
  solutions       : List (String × String)
  mesh_tags       : List (String × String)
  entity_maps     : List (String × String × String)
  active_mesh     : Option String
  deriving Repr

end DolfinxProofs
