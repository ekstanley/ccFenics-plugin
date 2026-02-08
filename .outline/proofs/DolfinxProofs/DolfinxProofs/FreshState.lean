import DolfinxProofs.Invariants
import DolfinxProofs.Operations

/-!
# Fresh State Validity

The empty session state trivially satisfies all 7 invariants
because all registries are empty and active_mesh is none.
-/

namespace DolfinxProofs

theorem freshState_valid : valid freshState := by
  unfold valid freshState
  simp [hasKey]

end DolfinxProofs
