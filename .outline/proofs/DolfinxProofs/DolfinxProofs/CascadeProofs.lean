import DolfinxProofs.Invariants
import DolfinxProofs.Operations
import DolfinxProofs.Basic

/-!
# Cascade Deletion Proofs

Proves that removeSpaceDeps and removeMesh preserve all 7 invariants.
removeMesh is the hardest proof in the project -- it requires showing
that cascade deletion of a mesh and all its dependents maintains
referential integrity.
-/

namespace DolfinxProofs

-- ========================================================================
-- Helper lemmas
-- ========================================================================

/-- If a pair is in a filtered list, its second element satisfies the filter. -/
private theorem filter_pair_snd {p : String × String → Bool}
    {k : String} {v : String} {l : List (String × String)}
    (h : (k, v) ∈ l.filter p) : p (k, v) = true :=
  (List.mem_filter.mp h).2

/-- If a pair is in a filtered list, it was in the original list. -/
private theorem mem_of_filter_pair {p : String × String → Bool}
    {k : String} {v : String} {l : List (String × String)}
    (h : (k, v) ∈ l.filter p) : (k, v) ∈ l :=
  (List.mem_filter.mp h).1

/-- hasKey on a filtered list: if sn is a key in filtered function_spaces,
    then there exists an entry (sn, mn') in the original that survived. -/
private theorem hasKey_of_filter {l : List (String × String)}
    {p : String × String → Bool} {sn : String}
    (h : hasKey (l.filter p) sn) :
    hasKey l sn := by
  unfold hasKey at *
  simp only [List.mem_map] at *
  obtain ⟨⟨k, v⟩, hmem, heq⟩ := h
  exact ⟨(k, v), (List.mem_filter.mp hmem).1, heq⟩

/-- Key lemma for INV-3 of removeMesh: if (sn, name) is NOT in fs,
    but sn IS a key in fs, then sn is still a key in fs filtered
    to entries whose second component is not name.

    Argument: sn is a key, so some (sn, b) is in fs.
    If b = name then (sn, name) in fs -- contradiction.
    So b != name and (sn, b) survives the filter. -/
private theorem key_survives_mesh_filter
    (fs : List (String × String)) (name : String) (sn : String)
    (h_key : hasKey fs sn)
    (h_not_dep : (sn, name) ∉ fs) :
    hasKey (fs.filter (fun (_, mn) => !decide (mn = name))) sn := by
  unfold hasKey at *
  simp only [List.mem_map] at *
  obtain ⟨⟨a, b⟩, hmem, heq⟩ := h_key
  simp at heq  -- heq : a = sn
  by_cases hbn : b = name
  · -- (a, b) = (sn, name) in fs, contradicting h_not_dep
    exfalso
    apply h_not_dep
    rw [heq.symm, hbn.symm]
    exact hmem
  · -- b != name, so (a, b) survives the filter
    exact ⟨(a, b), List.mem_filter.mpr ⟨hmem, by simp [hbn]⟩, heq⟩

-- ========================================================================
-- T9: removeSpaceDeps preserves invariants
-- ========================================================================

/-- Removing space dependents preserves all 7 invariants.
    Only functions/bcs/solutions are modified (filtered).
    function_spaces, meshes, etc. are unchanged. -/
theorem removeSpaceDeps_valid (s : SessionState) (spaceName : String)
    (h : valid s) : valid (removeSpaceDeps s spaceName) := by
  obtain ⟨h1, h2, h3, h4, h5, h6, h7⟩ := h
  unfold valid removeSpaceDeps
  simp only
  refine ⟨h1, h2, ?_, ?_, ?_, h6, h7⟩
  · -- INV-3: remaining functions still reference valid function_spaces
    intro k sn hmem
    exact h3 k sn (mem_of_filter_pair hmem)
  · -- INV-4: remaining BCs still reference valid function_spaces
    intro k sn hmem
    exact h4 k sn (mem_of_filter_pair hmem)
  · -- INV-5: remaining solutions still reference valid function_spaces
    intro k sn hmem
    exact h5 k sn (mem_of_filter_pair hmem)

-- ========================================================================
-- T10: removeMesh preserves invariants [MAIN THEOREM]
-- ========================================================================

/-- Cascade deletion of a mesh preserves all 7 referential integrity invariants.
    This is the highest-value proof, verifying that session.py L514-573
    correctly maintains referential integrity. -/
theorem removeMesh_valid (s : SessionState) (name : String)
    (h : valid s) : valid (removeMesh s name) := by
  obtain ⟨h1, h2, h3, h4, h5, h6, h7⟩ := h
  unfold valid removeMesh
  simp only
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_⟩

  · -- INV-1: active_mesh
    intro m hm
    split at hm
    · contradiction
    · -- active_mesh unchanged, and m != name (since active_mesh != some name)
      rename_i hne
      have hm_in := h1 m hm
      -- m is in old meshes, need to show m survives filter
      have hm_ne : m ≠ name := by
        intro heq; subst heq; exact hne hm
      exact List.mem_filter.mpr ⟨hm_in, by simp [hm_ne]⟩

  · -- INV-2: function_spaces -> meshes
    intro k mn hmem
    have ⟨hmem_orig, hfilt⟩ := List.mem_filter.mp hmem
    -- mn != name (from filter condition)
    simp at hfilt
    -- mn was in old meshes (by old INV-2)
    have hmn_in := h2 k mn hmem_orig
    -- mn survives the meshes filter since mn != name
    exact List.mem_filter.mpr ⟨hmn_in, by simp [hfilt]⟩

  · -- INV-3: functions -> function_spaces (the hard case)
    intro k sn hmem
    -- (k, sn) survived the functions filter, so sn is not in depSpaceKeys
    have ⟨hmem_orig, hfilt⟩ := List.mem_filter.mp hmem
    -- sn was a valid function_space key (old INV-3)
    have h_sn_key := h3 k sn hmem_orig
    -- sn not in depSpaceKeys (from filter condition)
    simp at hfilt
    -- Apply the key lemma
    exact key_survives_mesh_filter s.function_spaces name sn h_sn_key hfilt

  · -- INV-4: bcs -> function_spaces (same argument as INV-3)
    intro k sn hmem
    have ⟨hmem_orig, hfilt⟩ := List.mem_filter.mp hmem
    have h_sn_key := h4 k sn hmem_orig
    simp at hfilt
    exact key_survives_mesh_filter s.function_spaces name sn h_sn_key hfilt

  · -- INV-5: solutions -> function_spaces (same argument as INV-3)
    intro k sn hmem
    have ⟨hmem_orig, hfilt⟩ := List.mem_filter.mp hmem
    have h_sn_key := h5 k sn hmem_orig
    simp at hfilt
    exact key_survives_mesh_filter s.function_spaces name sn h_sn_key hfilt

  · -- INV-6: mesh_tags -> meshes (same argument as INV-2)
    intro k mn hmem
    have ⟨hmem_orig, hfilt⟩ := List.mem_filter.mp hmem
    simp at hfilt
    have hmn_in := h6 k mn hmem_orig
    exact List.mem_filter.mpr ⟨hmn_in, by simp [hfilt]⟩

  · -- INV-7: entity_maps -> meshes
    intro k pm cm hmem
    have ⟨hmem_orig, hfilt⟩ := List.mem_filter.mp hmem
    simp at hfilt
    obtain ⟨hpm_ne, hcm_ne⟩ := hfilt
    have ⟨hpm_in, hcm_in⟩ := h7 k pm cm hmem_orig
    exact ⟨List.mem_filter.mpr ⟨hpm_in, by simp [hpm_ne]⟩,
           List.mem_filter.mpr ⟨hcm_in, by simp [hcm_ne]⟩⟩

end DolfinxProofs
