import DolfinxProofs.Invariants
import DolfinxProofs.Operations
import DolfinxProofs.FreshState

/-!
# Cleanup Validity

cleanup produces freshState, which is already proven valid (all 8 invariants).
-/

namespace DolfinxProofs

theorem cleanup_valid (s : SessionState) : valid (cleanup s) :=
  freshState_valid

end DolfinxProofs
