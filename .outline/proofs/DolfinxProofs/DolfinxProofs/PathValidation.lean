/-!
# Path Validation Security Theorem

Models `_validate_output_path()` from postprocess.py lines 66-85.
Both `resolve` and `check` are abstract -- we prove the validation
logic is correct regardless of their implementation.
-/

namespace DolfinxProofs

/-- Result of path validation: either a valid path or an error. -/
inductive PathResult where
  | ok    : String → PathResult
  | error : String → PathResult
  deriving Repr

/-- Abstract validation function.
    `resolve` models `os.path.realpath . os.path.join("/workspace", .)`
    `check` models `str.startswith("/workspace")` -/
def validateOutputPath (resolve : String → String) (check : String → Bool)
    (path : String) : PathResult :=
  if check (resolve path) then
    PathResult.ok (resolve path)
  else
    PathResult.error path

/-- CONTAINMENT: If validation returns Ok, the check passes. -/
theorem validateOutputPath_ok_check (resolve : String → String)
    (check : String → Bool) (path : String) (resolved : String)
    (h : validateOutputPath resolve check path = PathResult.ok resolved) :
    check resolved = true := by
  unfold validateOutputPath at h
  split at h
  · next hc =>
    have heq : resolved = resolve path := (PathResult.ok.inj h).symm
    rw [heq]; exact hc
  · contradiction

/-- SECURITY THEOREM: validateOutputPath returns Ok only for checked paths. -/
theorem validateOutputPath_safe (resolve : String → String)
    (check : String → Bool) (path : String) :
    (∃ (resolved : String),
        validateOutputPath resolve check path = PathResult.ok resolved
        ∧ check resolved = true)
    ∨
    (∃ (msg : String), validateOutputPath resolve check path = PathResult.error msg) := by
  unfold validateOutputPath
  split
  · next hc => left; exact ⟨resolve path, rfl, hc⟩
  · right; exact ⟨path, rfl⟩

/-- TOTALITY: validateOutputPath always produces a result. -/
theorem validateOutputPath_total (resolve : String → String)
    (check : String → Bool) (path : String) :
    ∃ (r : PathResult), validateOutputPath resolve check path = r :=
  ⟨_, rfl⟩

end DolfinxProofs
