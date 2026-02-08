import Lake
open Lake DSL

package DolfinxProofs where
  leanOptions := #[
    ⟨`autoImplicit, false⟩
  ]

@[default_target]
lean_lib DolfinxProofs where
  srcDir := "."
