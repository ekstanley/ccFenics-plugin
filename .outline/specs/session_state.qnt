// session_state.qnt -- DOLFINx MCP SessionState complete specification
//
// Single-file Quint spec containing:
//   1. State machine model (12 vars, 12 actions)
//   2. 8 referential integrity invariants (INV-1 through INV-8)
//   3. Safety/liveness properties
//   4. Concrete model-check scenarios
//
// Corresponds to src/dolfinx_mcp/session.py:263-342
//
// Bounded name pools for exhaustive model checking:
//   meshes:  {m1, m2, m3}
//   spaces:  {V1, V2}
//   funcs:   {f1, f2}
//   bcs:     {bc1, bc2}
//   sols:    {u1, u2}
//   tags:    {t1, t2}
//   emaps:   {em1, em2}
//   forms:   {F1, F2}
//   ufls:    {sym1, sym2}

module session_state {

  // =========================================================================
  // State variables (corresponds to SessionState.__init__)
  // =========================================================================

  var meshes:          Set[str]
  var function_spaces: str -> str           // space_name -> mesh_name
  var functions:       str -> str           // func_name  -> space_name
  var bcs:             str -> str           // bc_name    -> space_name
  var solutions:       str -> str           // sol_name   -> space_name
  var mesh_tags:       str -> str           // tag_name   -> mesh_name
  var entity_maps:     str -> (str, str)    // emap_name  -> (parent, child)
  var active_mesh:     str                  // "" = None sentinel

  // Key-tracking sets (Quint maps don't enumerate keys directly)
  var space_keys: Set[str]
  var func_keys:  Set[str]
  var bc_keys:    Set[str]
  var sol_keys:   Set[str]
  var tag_keys:   Set[str]
  var emap_keys:  Set[str]
  var form_keys:         Set[str]
  var ufl_symbol_keys:   Set[str]
  var solver_diag_count: int
  var log_buffer_count:  int

  // =========================================================================
  // Name pools (bounded for exhaustive checking)
  // =========================================================================

  val MESH_NAMES:  Set[str] = Set("m1", "m2", "m3")
  val SPACE_NAMES: Set[str] = Set("V1", "V2")
  val FUNC_NAMES:  Set[str] = Set("f1", "f2")
  val BC_NAMES:    Set[str] = Set("bc1", "bc2")
  val SOL_NAMES:   Set[str] = Set("u1", "u2")
  val TAG_NAMES:   Set[str] = Set("t1", "t2")
  val EMAP_NAMES:  Set[str] = Set("em1", "em2")
  val FORM_NAMES: Set[str] = Set("F1", "F2")
  val UFL_NAMES:  Set[str] = Set("sym1", "sym2")

  // =========================================================================
  // Init
  // =========================================================================

  action init = all {
    meshes'          = Set(),
    function_spaces' = Map(),
    functions'       = Map(),
    bcs'             = Map(),
    solutions'       = Map(),
    mesh_tags'       = Map(),
    entity_maps'     = Map(),
    active_mesh'     = "",
    space_keys'      = Set(),
    func_keys'       = Set(),
    bc_keys'         = Set(),
    sol_keys'        = Set(),
    tag_keys'        = Set(),
    emap_keys'       = Set(),
    form_keys'         = Set(),
    ufl_symbol_keys'   = Set(),
    solver_diag_count' = 0,
    log_buffer_count'  = 0,
  }

  // =========================================================================
  // Actions (10 total)
  // =========================================================================

  action registerMesh(name: str): bool = all {
    not(meshes.contains(name)),
    meshes'          = meshes.union(Set(name)),
    active_mesh'     = name,
    function_spaces' = function_spaces,
    functions'       = functions,
    bcs'             = bcs,
    solutions'       = solutions,
    mesh_tags'       = mesh_tags,
    entity_maps'     = entity_maps,
    space_keys'      = space_keys,
    func_keys'       = func_keys,
    bc_keys'         = bc_keys,
    sol_keys'        = sol_keys,
    tag_keys'        = tag_keys,
    emap_keys'       = emap_keys,
    form_keys'         = form_keys,
    ufl_symbol_keys'   = ufl_symbol_keys,
    solver_diag_count' = solver_diag_count,
    log_buffer_count'  = log_buffer_count,
  }

  action registerFunctionSpace(name: str, meshName: str): bool = all {
    meshes.contains(meshName),
    not(space_keys.contains(name)),
    function_spaces' = function_spaces.put(name, meshName),
    space_keys'      = space_keys.union(Set(name)),
    meshes'          = meshes,
    active_mesh'     = active_mesh,
    functions'       = functions,
    bcs'             = bcs,
    solutions'       = solutions,
    mesh_tags'       = mesh_tags,
    entity_maps'     = entity_maps,
    func_keys'       = func_keys,
    bc_keys'         = bc_keys,
    sol_keys'        = sol_keys,
    tag_keys'        = tag_keys,
    emap_keys'       = emap_keys,
    form_keys'         = form_keys,
    ufl_symbol_keys'   = ufl_symbol_keys,
    solver_diag_count' = solver_diag_count,
    log_buffer_count'  = log_buffer_count,
  }

  action registerFunction(name: str, spaceName: str): bool = all {
    space_keys.contains(spaceName),
    not(func_keys.contains(name)),
    functions'       = functions.put(name, spaceName),
    func_keys'       = func_keys.union(Set(name)),
    meshes'          = meshes,
    active_mesh'     = active_mesh,
    function_spaces' = function_spaces,
    bcs'             = bcs,
    solutions'       = solutions,
    mesh_tags'       = mesh_tags,
    entity_maps'     = entity_maps,
    space_keys'      = space_keys,
    bc_keys'         = bc_keys,
    sol_keys'        = sol_keys,
    tag_keys'        = tag_keys,
    emap_keys'       = emap_keys,
    form_keys'         = form_keys,
    ufl_symbol_keys'   = ufl_symbol_keys,
    solver_diag_count' = solver_diag_count,
    log_buffer_count'  = log_buffer_count,
  }

  action registerBC(name: str, spaceName: str): bool = all {
    space_keys.contains(spaceName),
    not(bc_keys.contains(name)),
    bcs'             = bcs.put(name, spaceName),
    bc_keys'         = bc_keys.union(Set(name)),
    meshes'          = meshes,
    active_mesh'     = active_mesh,
    function_spaces' = function_spaces,
    functions'       = functions,
    solutions'       = solutions,
    mesh_tags'       = mesh_tags,
    entity_maps'     = entity_maps,
    space_keys'      = space_keys,
    func_keys'       = func_keys,
    sol_keys'        = sol_keys,
    tag_keys'        = tag_keys,
    emap_keys'       = emap_keys,
    form_keys'         = form_keys,
    ufl_symbol_keys'   = ufl_symbol_keys,
    solver_diag_count' = solver_diag_count,
    log_buffer_count'  = log_buffer_count,
  }

  action registerSolution(name: str, spaceName: str): bool = all {
    space_keys.contains(spaceName),
    not(sol_keys.contains(name)),
    solutions'       = solutions.put(name, spaceName),
    sol_keys'        = sol_keys.union(Set(name)),
    meshes'          = meshes,
    active_mesh'     = active_mesh,
    function_spaces' = function_spaces,
    functions'       = functions,
    bcs'             = bcs,
    mesh_tags'       = mesh_tags,
    entity_maps'     = entity_maps,
    space_keys'      = space_keys,
    func_keys'       = func_keys,
    bc_keys'         = bc_keys,
    tag_keys'        = tag_keys,
    emap_keys'       = emap_keys,
    form_keys'         = form_keys,
    ufl_symbol_keys'   = ufl_symbol_keys,
    solver_diag_count' = solver_diag_count,
    log_buffer_count'  = log_buffer_count,
  }

  action registerMeshTags(name: str, meshName: str): bool = all {
    meshes.contains(meshName),
    not(tag_keys.contains(name)),
    mesh_tags'       = mesh_tags.put(name, meshName),
    tag_keys'        = tag_keys.union(Set(name)),
    meshes'          = meshes,
    active_mesh'     = active_mesh,
    function_spaces' = function_spaces,
    functions'       = functions,
    bcs'             = bcs,
    solutions'       = solutions,
    entity_maps'     = entity_maps,
    space_keys'      = space_keys,
    func_keys'       = func_keys,
    bc_keys'         = bc_keys,
    sol_keys'        = sol_keys,
    emap_keys'       = emap_keys,
    form_keys'         = form_keys,
    ufl_symbol_keys'   = ufl_symbol_keys,
    solver_diag_count' = solver_diag_count,
    log_buffer_count'  = log_buffer_count,
  }

  action registerEntityMap(name: str, parent: str, child: str): bool = all {
    meshes.contains(parent),
    meshes.contains(child),
    not(emap_keys.contains(name)),
    entity_maps'     = entity_maps.put(name, (parent, child)),
    emap_keys'       = emap_keys.union(Set(name)),
    meshes'          = meshes,
    active_mesh'     = active_mesh,
    function_spaces' = function_spaces,
    functions'       = functions,
    bcs'             = bcs,
    solutions'       = solutions,
    mesh_tags'       = mesh_tags,
    space_keys'      = space_keys,
    func_keys'       = func_keys,
    bc_keys'         = bc_keys,
    sol_keys'        = sol_keys,
    tag_keys'        = tag_keys,
    form_keys'         = form_keys,
    ufl_symbol_keys'   = ufl_symbol_keys,
    solver_diag_count' = solver_diag_count,
    log_buffer_count'  = log_buffer_count,
  }

  // registerForm: forms have no FK but require at least one space to exist
  action registerForm(name: str): bool = all {
    space_keys.size() > 0,
    not(form_keys.contains(name)),
    form_keys'         = form_keys.union(Set(name)),
    meshes'            = meshes,
    active_mesh'       = active_mesh,
    function_spaces'   = function_spaces,
    functions'         = functions,
    bcs'               = bcs,
    solutions'         = solutions,
    mesh_tags'         = mesh_tags,
    entity_maps'       = entity_maps,
    space_keys'        = space_keys,
    func_keys'         = func_keys,
    bc_keys'           = bc_keys,
    sol_keys'          = sol_keys,
    tag_keys'          = tag_keys,
    emap_keys'         = emap_keys,
    ufl_symbol_keys'   = ufl_symbol_keys,
    solver_diag_count' = solver_diag_count,
    log_buffer_count'  = log_buffer_count,
  }

  // registerUflSymbol: same precondition as forms
  action registerUflSymbol(name: str): bool = all {
    space_keys.size() > 0,
    not(ufl_symbol_keys.contains(name)),
    ufl_symbol_keys'   = ufl_symbol_keys.union(Set(name)),
    meshes'            = meshes,
    active_mesh'       = active_mesh,
    function_spaces'   = function_spaces,
    functions'         = functions,
    bcs'               = bcs,
    solutions'         = solutions,
    mesh_tags'         = mesh_tags,
    entity_maps'       = entity_maps,
    space_keys'        = space_keys,
    func_keys'         = func_keys,
    bc_keys'           = bc_keys,
    sol_keys'          = sol_keys,
    tag_keys'          = tag_keys,
    emap_keys'         = emap_keys,
    form_keys'         = form_keys,
    solver_diag_count' = solver_diag_count,
    log_buffer_count'  = log_buffer_count,
  }

  // removeMesh: cascade delete all dependents
  // Corresponds to SessionState.remove_mesh + _remove_space_dependents
  action removeMesh(name: str): bool = all {
    meshes.contains(name),

    val deadSpaces = space_keys.filter(s => function_spaces.get(s) == name)
    val deadFuncs  = func_keys.filter(f => deadSpaces.contains(functions.get(f)))
    val deadBCs    = bc_keys.filter(b => deadSpaces.contains(bcs.get(b)))
    val deadSols   = sol_keys.filter(u => deadSpaces.contains(solutions.get(u)))
    val deadTags   = tag_keys.filter(t => mesh_tags.get(t) == name)
    val deadEmaps  = emap_keys.filter(e =>
      entity_maps.get(e)._1 == name or entity_maps.get(e)._2 == name
    )

    all {
      meshes'          = meshes.exclude(Set(name)),
      active_mesh'     = if (active_mesh == name) "" else active_mesh,
      space_keys'      = space_keys.exclude(deadSpaces),
      func_keys'       = func_keys.exclude(deadFuncs),
      bc_keys'         = bc_keys.exclude(deadBCs),
      sol_keys'        = sol_keys.exclude(deadSols),
      tag_keys'        = tag_keys.exclude(deadTags),
      emap_keys'       = emap_keys.exclude(deadEmaps),
      // Maps: orphaned entries unreachable via key sets
      function_spaces' = function_spaces,
      functions'       = functions,
      bcs'             = bcs,
      solutions'       = solutions,
      mesh_tags'       = mesh_tags,
      entity_maps'     = entity_maps,
      // Forms and UFL symbols cleared wholesale when any space is deleted
      form_keys'         = if (deadSpaces.size() > 0) Set() else form_keys,
      ufl_symbol_keys'   = if (deadSpaces.size() > 0) Set() else ufl_symbol_keys,
      solver_diag_count' = solver_diag_count,
      log_buffer_count'  = log_buffer_count,
    }
  }

  // cleanup: reset to empty state
  action cleanup: bool = all {
    meshes'          = Set(),
    function_spaces' = Map(),
    functions'       = Map(),
    bcs'             = Map(),
    solutions'       = Map(),
    mesh_tags'       = Map(),
    entity_maps'     = Map(),
    active_mesh'     = "",
    space_keys'      = Set(),
    func_keys'       = Set(),
    bc_keys'         = Set(),
    sol_keys'        = Set(),
    tag_keys'        = Set(),
    emap_keys'       = Set(),
    form_keys'         = Set(),
    ufl_symbol_keys'   = Set(),
    solver_diag_count' = 0,
    log_buffer_count'  = 0,
  }

  // No-op stutter step (for scenario terminal assertions)
  action stutter: bool = all {
    meshes'          = meshes,
    function_spaces' = function_spaces,
    functions'       = functions,
    bcs'             = bcs,
    solutions'       = solutions,
    mesh_tags'       = mesh_tags,
    entity_maps'     = entity_maps,
    active_mesh'     = active_mesh,
    space_keys'      = space_keys,
    func_keys'       = func_keys,
    bc_keys'         = bc_keys,
    sol_keys'        = sol_keys,
    tag_keys'        = tag_keys,
    emap_keys'       = emap_keys,
    form_keys'         = form_keys,
    ufl_symbol_keys'   = ufl_symbol_keys,
    solver_diag_count' = solver_diag_count,
    log_buffer_count'  = log_buffer_count,
  }

  // =========================================================================
  // Non-deterministic step
  // =========================================================================

  action step = any {
    nondet m = MESH_NAMES.oneOf(); registerMesh(m),
    nondet s = SPACE_NAMES.oneOf(); nondet m = MESH_NAMES.oneOf();
      registerFunctionSpace(s, m),
    nondet f = FUNC_NAMES.oneOf(); nondet s = SPACE_NAMES.oneOf();
      registerFunction(f, s),
    nondet b = BC_NAMES.oneOf(); nondet s = SPACE_NAMES.oneOf();
      registerBC(b, s),
    nondet u = SOL_NAMES.oneOf(); nondet s = SPACE_NAMES.oneOf();
      registerSolution(u, s),
    nondet t = TAG_NAMES.oneOf(); nondet m = MESH_NAMES.oneOf();
      registerMeshTags(t, m),
    nondet e = EMAP_NAMES.oneOf(); nondet p = MESH_NAMES.oneOf();
      nondet c = MESH_NAMES.oneOf(); registerEntityMap(e, p, c),
    nondet m = MESH_NAMES.oneOf(); removeMesh(m),
    nondet fn = FORM_NAMES.oneOf(); registerForm(fn),
    nondet un = UFL_NAMES.oneOf(); registerUflSymbol(un),
    cleanup,
  }

  // =========================================================================
  // Invariants (INV-1 through INV-7, session.py:285-342)
  // =========================================================================

  // INV-1: active_mesh must be "" (None) or a valid mesh key
  val inv1_active_mesh_valid: bool =
    active_mesh == "" or meshes.contains(active_mesh)

  // INV-2: all function_spaces reference valid meshes
  val inv2_spaces_reference_valid_meshes: bool =
    space_keys.forall(s => meshes.contains(function_spaces.get(s)))

  // INV-3: all functions reference valid function_spaces
  val inv3_functions_reference_valid_spaces: bool =
    func_keys.forall(f => space_keys.contains(functions.get(f)))

  // INV-4: all BCs reference valid function_spaces
  val inv4_bcs_reference_valid_spaces: bool =
    bc_keys.forall(b => space_keys.contains(bcs.get(b)))

  // INV-5: all solutions reference valid function_spaces
  val inv5_solutions_reference_valid_spaces: bool =
    sol_keys.forall(u => space_keys.contains(solutions.get(u)))

  // INV-6: all mesh_tags reference valid meshes
  val inv6_tags_reference_valid_meshes: bool =
    tag_keys.forall(t => meshes.contains(mesh_tags.get(t)))

  // INV-7: all entity_maps reference valid meshes (both parent and child)
  val inv7_entity_maps_reference_valid_meshes: bool =
    emap_keys.forall(e =>
      meshes.contains(entity_maps.get(e)._1) and
      meshes.contains(entity_maps.get(e)._2)
    )

  // INV-8: forms can only exist if at least one function_space exists
  // (forms are cleared wholesale when spaces are deleted, so
  //  forms_non_empty => spaces_non_empty)
  val inv8_forms_require_spaces: bool =
    form_keys.size() > 0 implies space_keys.size() > 0

  // Composite: all 8 invariants
  val all_invariants: bool = and {
    inv1_active_mesh_valid,
    inv2_spaces_reference_valid_meshes,
    inv3_functions_reference_valid_spaces,
    inv4_bcs_reference_valid_spaces,
    inv5_solutions_reference_valid_spaces,
    inv6_tags_reference_valid_meshes,
    inv7_entity_maps_reference_valid_meshes,
    inv8_forms_require_spaces,
  }

  // =========================================================================
  // Safety property
  // =========================================================================

  val is_empty_state: bool = and {
    meshes      == Set(),
    space_keys  == Set(),
    func_keys   == Set(),
    bc_keys     == Set(),
    sol_keys    == Set(),
    tag_keys    == Set(),
    emap_keys   == Set(),
    form_keys   == Set(),
    ufl_symbol_keys == Set(),
    active_mesh == "",
  }

  // =========================================================================
  // Scenarios
  // =========================================================================

  // Scenario 1: Full cascade deletion
  run cascade_scenario = {
    init
      .then(registerMesh("m1"))
      .then(registerFunctionSpace("V1", "m1"))
      .then(registerFunction("f1", "V1"))
      .then(registerBC("bc1", "V1"))
      .then(registerSolution("u1", "V1"))
      .then(registerMeshTags("t1", "m1"))
      .then(removeMesh("m1"))
      .then(all {
        assert(not(meshes.contains("m1"))),
        assert(not(space_keys.contains("V1"))),
        assert(not(func_keys.contains("f1"))),
        assert(not(bc_keys.contains("bc1"))),
        assert(not(sol_keys.contains("u1"))),
        assert(not(tag_keys.contains("t1"))),
        assert(is_empty_state),
        assert(all_invariants),
        stutter,
      })
  }

  // Scenario 2: Partial removal -- remove one mesh, other survives
  run partial_remove_scenario = {
    init
      .then(registerMesh("m1"))
      .then(registerMesh("m2"))
      .then(registerFunctionSpace("V1", "m1"))
      .then(registerFunctionSpace("V2", "m2"))
      .then(registerFunction("f1", "V1"))
      .then(registerFunction("f2", "V2"))
      .then(removeMesh("m1"))
      .then(all {
        assert(not(meshes.contains("m1"))),
        assert(not(space_keys.contains("V1"))),
        assert(not(func_keys.contains("f1"))),
        assert(meshes.contains("m2")),
        assert(space_keys.contains("V2")),
        assert(func_keys.contains("f2")),
        assert(all_invariants),
        stutter,
      })
  }

  // Scenario 3: Entity map with shared parent/child
  run entity_map_shared_mesh = {
    init
      .then(registerMesh("m1"))
      .then(registerEntityMap("em1", "m1", "m1"))
      .then(removeMesh("m1"))
      .then(all {
        assert(not(meshes.contains("m1"))),
        assert(not(emap_keys.contains("em1"))),
        assert(is_empty_state),
        assert(all_invariants),
        stutter,
      })
  }

  // Scenario 4: Cleanup after complex state
  run cleanup_complex_scenario = {
    init
      .then(registerMesh("m1"))
      .then(registerMesh("m2"))
      .then(registerFunctionSpace("V1", "m1"))
      .then(registerFunctionSpace("V2", "m2"))
      .then(registerFunction("f1", "V1"))
      .then(registerFunction("f2", "V2"))
      .then(registerBC("bc1", "V1"))
      .then(registerBC("bc2", "V2"))
      .then(registerSolution("u1", "V1"))
      .then(registerSolution("u2", "V2"))
      .then(registerMeshTags("t1", "m1"))
      .then(registerMeshTags("t2", "m2"))
      .then(registerEntityMap("em1", "m1", "m2"))
      .then(cleanup)
      .then(all {
        assert(is_empty_state),
        assert(all_invariants),
        stutter,
      })
  }

  // Scenario 5: Entity map remove parent only
  run entity_map_remove_parent = {
    init
      .then(registerMesh("m1"))
      .then(registerMesh("m2"))
      .then(registerEntityMap("em1", "m1", "m2"))
      .then(registerFunctionSpace("V2", "m2"))
      .then(removeMesh("m1"))
      .then(all {
        assert(not(meshes.contains("m1"))),
        assert(not(emap_keys.contains("em1"))),
        assert(meshes.contains("m2")),
        assert(space_keys.contains("V2")),
        assert(all_invariants),
        stutter,
      })
  }

  // Scenario 6: Forms cascade -- forms cleared when spaces deleted
  run forms_cascade_scenario = {
    init
      .then(registerMesh("m1"))
      .then(registerFunctionSpace("V1", "m1"))
      .then(registerForm("F1"))
      .then(registerUflSymbol("sym1"))
      .then(removeMesh("m1"))
      .then(all {
        assert(not(meshes.contains("m1"))),
        assert(not(space_keys.contains("V1"))),
        assert(not(form_keys.contains("F1"))),
        assert(not(ufl_symbol_keys.contains("sym1"))),
        assert(is_empty_state),
        assert(all_invariants),
        stutter,
      })
  }

}
