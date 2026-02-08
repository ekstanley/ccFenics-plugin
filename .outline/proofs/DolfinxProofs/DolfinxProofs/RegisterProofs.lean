import DolfinxProofs.Invariants
import DolfinxProofs.Operations
import DolfinxProofs.Basic

/-!
# Registration Preservation Theorems

Each register operation preserves all 8 referential integrity invariants,
given the appropriate precondition (e.g., mesh_name must already exist).
-/

namespace DolfinxProofs

/-- T2: Registering a mesh preserves all 8 invariants. -/
theorem registerMesh_valid (s : SessionState) (name : String)
    (h : valid s) : valid (registerMesh s name) := by
  obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h
  unfold valid registerMesh
  simp only
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · -- INV-1: active_mesh = some name, and name ∈ name :: meshes
    intro m hm
    simp at hm
    subst hm
    exact List.mem_cons_self name s.meshes
  · -- INV-2: function_spaces unchanged, mesh_name still valid (weakened by cons)
    intro k mn hmem
    exact List.mem_cons_of_mem name (h2 k mn hmem)
  · -- INV-3: functions unchanged, function_spaces unchanged
    exact h3
  · -- INV-4: bcs unchanged
    exact h4
  · -- INV-5: solutions unchanged
    exact h5
  · -- INV-6: mesh_tags unchanged, meshes weakened
    intro k mn hmem
    exact List.mem_cons_of_mem name (h6 k mn hmem)
  · -- INV-7: entity_maps unchanged, meshes weakened
    intro k pm cm hmem
    have ⟨hp, hc⟩ := h7 k pm cm hmem
    exact ⟨List.mem_cons_of_mem name hp, List.mem_cons_of_mem name hc⟩
  · -- INV-8: forms unchanged, function_spaces unchanged
    exact h8

/-- T3: Registering a function space preserves invariants. -/
theorem registerFunctionSpace_valid (s : SessionState) (name : String) (meshName : String)
    (h : valid s) (hpre : meshName ∈ s.meshes) :
    valid (registerFunctionSpace s name meshName) := by
  obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h
  unfold valid registerFunctionSpace
  simp only
  refine ⟨h1, ?_, ?_, ?_, ?_, h6, h7, ?_⟩
  · -- INV-2: new entry (name, meshName) satisfies by hpre; old entries by h2
    intro k mn hmem
    simp at hmem
    rcases hmem with ⟨rfl, rfl⟩ | hmem
    · exact hpre
    · exact h2 k mn hmem
  · -- INV-3: hasKey is monotone under cons
    intro k sn hmem
    exact hasKey_cons _ _ (name, meshName) (h3 k sn hmem)
  · -- INV-4: same monotonicity
    intro k sn hmem
    exact hasKey_cons _ _ (name, meshName) (h4 k sn hmem)
  · -- INV-5: same monotonicity
    intro k sn hmem
    exact hasKey_cons _ _ (name, meshName) (h5 k sn hmem)
  · -- INV-8: forms unchanged, function_spaces grows (cons is non-empty)
    intro _
    exact List.cons_ne_nil _ _

/-- T4: Registering a function preserves invariants. -/
theorem registerFunction_valid (s : SessionState) (name : String) (spaceName : String)
    (h : valid s) (hpre : hasKey s.function_spaces spaceName) :
    valid (registerFunction s name spaceName) := by
  obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h
  unfold valid registerFunction
  simp only
  refine ⟨h1, h2, ?_, h4, h5, h6, h7, h8⟩
  intro k sn hmem
  simp at hmem
  rcases hmem with ⟨rfl, rfl⟩ | hmem
  · exact hpre
  · exact h3 k sn hmem

/-- T5: Registering a BC preserves invariants. -/
theorem registerBC_valid (s : SessionState) (name : String) (spaceName : String)
    (h : valid s) (hpre : hasKey s.function_spaces spaceName) :
    valid (registerBC s name spaceName) := by
  obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h
  unfold valid registerBC
  simp only
  refine ⟨h1, h2, h3, ?_, h5, h6, h7, h8⟩
  intro k sn hmem
  simp at hmem
  rcases hmem with ⟨rfl, rfl⟩ | hmem
  · exact hpre
  · exact h4 k sn hmem

/-- T6: Registering a solution preserves invariants. -/
theorem registerSolution_valid (s : SessionState) (name : String) (spaceName : String)
    (h : valid s) (hpre : hasKey s.function_spaces spaceName) :
    valid (registerSolution s name spaceName) := by
  obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h
  unfold valid registerSolution
  simp only
  refine ⟨h1, h2, h3, h4, ?_, h6, h7, h8⟩
  intro k sn hmem
  simp at hmem
  rcases hmem with ⟨rfl, rfl⟩ | hmem
  · exact hpre
  · exact h5 k sn hmem

/-- T7: Registering mesh tags preserves invariants. -/
theorem registerMeshTags_valid (s : SessionState) (name : String) (meshName : String)
    (h : valid s) (hpre : meshName ∈ s.meshes) :
    valid (registerMeshTags s name meshName) := by
  obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h
  unfold valid registerMeshTags
  simp only
  refine ⟨h1, h2, h3, h4, h5, ?_, h7, h8⟩
  intro k mn hmem
  simp at hmem
  rcases hmem with ⟨rfl, rfl⟩ | hmem
  · exact hpre
  · exact h6 k mn hmem

/-- T8: Registering an entity map preserves invariants. -/
theorem registerEntityMap_valid (s : SessionState) (name : String)
    (parentMesh : String) (childMesh : String)
    (h : valid s) (hp : parentMesh ∈ s.meshes) (hc : childMesh ∈ s.meshes) :
    valid (registerEntityMap s name parentMesh childMesh) := by
  obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ := h
  unfold valid registerEntityMap
  simp only
  refine ⟨h1, h2, h3, h4, h5, h6, ?_, h8⟩
  intro k pm cm hmem
  simp at hmem
  rcases hmem with ⟨rfl, rfl, rfl⟩ | hmem
  · exact ⟨hp, hc⟩
  · exact h7 k pm cm hmem

end DolfinxProofs
