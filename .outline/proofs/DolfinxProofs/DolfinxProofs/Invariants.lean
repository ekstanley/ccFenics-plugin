import DolfinxProofs.State
import DolfinxProofs.Basic

/-!
# Referential Integrity Invariants

Direct Lean translation of the 7 invariants checked by
`SessionState.check_invariants()` in session.py lines 285-342.
-/

namespace DolfinxProofs

/-- All 7 referential integrity invariants as a single predicate.
    Matches the Python `check_invariants()` method exactly. -/
def valid (s : SessionState) : Prop :=
  -- INV-1: active_mesh is None or references a valid mesh key
  (∀ (m : String), s.active_mesh = some m → m ∈ s.meshes)
  ∧
  -- INV-2: all function_spaces reference valid mesh keys
  (∀ (k : String) (mn : String), (k, mn) ∈ s.function_spaces → mn ∈ s.meshes)
  ∧
  -- INV-3: all functions reference valid function_space keys
  (∀ (k : String) (sn : String), (k, sn) ∈ s.functions → hasKey s.function_spaces sn)
  ∧
  -- INV-4: all BCs reference valid function_space keys
  (∀ (k : String) (sn : String), (k, sn) ∈ s.bcs → hasKey s.function_spaces sn)
  ∧
  -- INV-5: all solutions reference valid function_space keys
  (∀ (k : String) (sn : String), (k, sn) ∈ s.solutions → hasKey s.function_spaces sn)
  ∧
  -- INV-6: all mesh_tags reference valid mesh keys
  (∀ (k : String) (mn : String), (k, mn) ∈ s.mesh_tags → mn ∈ s.meshes)
  ∧
  -- INV-7: all entity_maps reference valid mesh keys (both parent and child)
  (∀ (k : String) (pm : String) (cm : String),
    (k, pm, cm) ∈ s.entity_maps → pm ∈ s.meshes ∧ cm ∈ s.meshes)

end DolfinxProofs
