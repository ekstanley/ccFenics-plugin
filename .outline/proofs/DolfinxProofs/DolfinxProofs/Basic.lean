/-!
# Basic Registry Definitions and Helper Lemmas

Minimal list-based model of Python dict registries.
Only foreign key relationships are tracked.
-/

namespace DolfinxProofs

/-- Check if a key exists in an association list (as first element of pair). -/
def hasKey {α : Type} (entries : List (String × α)) (k : String) : Prop :=
  k ∈ entries.map Prod.fst

/-- hasKey is monotone: adding an entry preserves existing keys. -/
theorem hasKey_cons {α : Type} (entries : List (String × α)) (k : String)
    (entry : String × α) (h : hasKey entries k) :
    hasKey (entry :: entries) k := by
  simp only [hasKey, List.map_cons]
  exact List.mem_cons_of_mem _ h

/-- A newly added key is in the association list. -/
theorem hasKey_cons_self {α : Type} (entries : List (String × α)) (k : String)
    (v : α) : hasKey ((k, v) :: entries) k := by
  simp only [hasKey, List.map_cons]
  exact List.mem_cons_self k _

/-- If an element survives a filter, it was in the original list. -/
theorem mem_of_mem_filter' {α : Type} {p : α → Bool} {x : α} {l : List α}
    (h : x ∈ l.filter p) : x ∈ l :=
  (List.mem_filter.mp h).1

/-- If an element is in a list and satisfies the predicate, it is in the filtered list. -/
theorem mem_filter_of_mem' {α : Type} {p : α → Bool} {x : α} {l : List α}
    (hmem : x ∈ l) (hp : p x = true) : x ∈ l.filter p :=
  List.mem_filter.mpr ⟨hmem, hp⟩

end DolfinxProofs
