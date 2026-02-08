/-!
# MeshInfo Data Structure Invariants

Models the __post_init__ checks from MeshInfo (session.py lines 42-54).
-/

namespace DolfinxProofs

/-- Minimal model of MeshInfo fields relevant to validation. -/
structure MeshInfoData where
  name         : String
  num_cells    : Nat
  num_vertices : Nat
  gdim         : Nat
  tdim         : Nat
  deriving Repr, DecidableEq

/-- MeshInfo is valid iff all __post_init__ checks pass. -/
def meshInfoValid (m : MeshInfoData) : Prop :=
  m.name ≠ ""
  ∧ m.num_cells > 0
  ∧ m.num_vertices > 0
  ∧ (m.gdim = 1 ∨ m.gdim = 2 ∨ m.gdim = 3)
  ∧ (m.tdim = 1 ∨ m.tdim = 2 ∨ m.tdim = 3)
  ∧ m.tdim ≤ m.gdim

/-- Example: a standard 2D triangular mesh is valid. -/
theorem example_2d_mesh_valid :
    meshInfoValid ⟨"unit_square", 200, 121, 2, 2⟩ := by
  refine ⟨by decide, by decide, by decide, Or.inr (Or.inl rfl), Or.inr (Or.inl rfl), by decide⟩

/-- tdim <= gdim: enumerate all 6 valid (gdim, tdim) pairs. -/
theorem valid_gdim_tdim_pairs (g : Nat) (t : Nat)
    (hg : g = 1 ∨ g = 2 ∨ g = 3) (ht : t = 1 ∨ t = 2 ∨ t = 3) (hle : t ≤ g) :
    (g, t) ∈ [(1,1), (2,1), (2,2), (3,1), (3,2), (3,3)] := by
  rcases hg with rfl | rfl | rfl <;> rcases ht with rfl | rfl | rfl <;>
    simp_all (config := { decide := true }) <;> omega

end DolfinxProofs
