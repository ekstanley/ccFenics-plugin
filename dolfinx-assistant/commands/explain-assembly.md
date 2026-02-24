# Command: /explain-assembly

Pedagogical walkthrough of how DOLFINx converts weak forms into linear systems.

## Usage

```
/explain-assembly
```

No arguments. Analyzes current session state and explains the assembly process for your problem.

## Workflow

### Step 1: Check Session State

Run:
```
get_session_state()
```

Look for:
- Active mesh and its properties (num_cells, num_vertices, cell_type)
- Function spaces defined (name, family, degree, num_dofs)
- Forms defined (bilinear and linear form expressions)
- Boundary conditions applied (which DOFs are constrained)

Output example:
```
Active mesh: unit_square_mesh (8×8 triangles, 81 vertices)
Function spaces:
  - V (Lagrange, degree 1, 81 DOFs)
Forms:
  - Bilinear: inner(grad(u), grad(v))*dx
  - Linear: f*v*dx
Boundary conditions: 1
  - BC on boundary (x[0] < 1e-14), value=0, 9 DOFs
```

### Step 2: Explain What Assembly Will Do

Print pedagogical explanation:

> Your Poisson problem will be assembled as follows:
>
> **Bilinear form** `inner(grad(u), grad(v))*dx` will produce a **stiffness matrix K**:
> - One 81×81 matrix element for each pair of basis function gradients
> - Only ~9 nonzeros per row (sparse, due to local support of basis functions)
> - Will capture how solutions at each node couple with neighbors
>
> **Linear form** `f*v*dx` will produce a **load vector F**:
> - One 81-element vector, one per DOF
> - Will encode your source term `f` integrated against each basis function
>
> **Boundary condition** at x=0 will apply "lifting":
> - 9 DOFs on the left boundary (x=0) will be constrained to u=0
> - RHS will be modified to account for this known value
> - Interior DOFs (remaining 72) will be solved for

### Step 3: Demonstrate with Actual Assembly

If forms are defined, run practical assembly operations:

**Assemble the matrix:**
```
assemble(target="matrix", form="inner(grad(u), grad(v))*dx", apply_bcs=True)
```

Expected output:
```
{
  "dims": [81, 81],
  "nnz": 451
}
```

Explain:
> **Stiffness matrix assembled:** 81×81 matrix with 451 nonzero entries.
> Average sparsity: 451/81 ≈ 5.6 nonzeros per row.
> This is typical for 2D triangular P1 elements (neighbors on unstructured mesh).

**Assemble the load vector:**
```
assemble(target="vector", form="f*v*dx", apply_bcs=True)
```

Expected output:
```
{
  "norm": 2.45,
  "size": 81
}
```

Explain:
> **Load vector assembled:** 81 components, L2 norm = 2.45.
> This vector encodes how your source term `f` integrates over each basis function region.

**Assemble a diagnostic scalar:**
```
assemble(target="scalar", form="f*f*dx")
```

Expected output:
```
{
  "value": 8.12
}
```

Explain:
> **Source term energy:** ∫ f² dx = 8.12.
> This tells us the L2 norm of the source is √8.12 ≈ 2.85.

### Step 4: Explain Boundary Condition Handling

If BCs are present:

Print:
> **Boundary condition application (lifting method):**
>
> You applied u=0 on boundary x[0]<1e-14 (9 DOFs).
>
> During assembly, DOLFINx will:
> 1. Create a "lifting" function u_D that equals 0 on the boundary
> 2. Solve for interior DOF values u_interior
> 3. Final solution = u_interior + u_D
>
> This preserves matrix symmetry while enforcing BCs.
> The left 9 rows of K will be zeroed, with diagonal = 1.
> The RHS F will be modified by subtracting K·u_D.

### Step 5: Explain the Solver Step

Print:
> **Linear system solve:**
>
> Once assembly is complete, you have:
> - **K** (81×81 stiffness matrix)
> - **F** (81-element load vector)
>
> The solver will compute: **u = K⁻¹ F**
>
> For this problem:
> - Direct solver (LU factorization) will factor K in ~0.001 seconds
> - Solution vector u will be 81 DOFs: the value at each mesh vertex
> - Boundary DOFs will be exactly 0 (from BC)
> - Interior DOFs will be computed to satisfy ∇²u ≈ f everywhere

### Step 6: (Optional) Pedagogical Poisson Example

If no forms are defined yet, offer to walk through a simple example:

> **Setting up a pedagogical Poisson problem:**
>
> Let me build a complete example step-by-step:
>
> ```python
> # 1. Create mesh
> create_unit_square(name="demo_mesh", nx=4, ny=4, cell_type="triangle")
>
> # 2. Create function space
> create_function_space(name="V", family="Lagrange", degree=1, mesh_name="demo_mesh")
>
> # 3. Define weak form
> define_variational_form(
>     bilinear="inner(grad(u), grad(v))*dx",
>     linear="f*v*dx"
> )
>
> # 4. Set source term
> set_material_properties(name="f", value="sin(pi*x[0])*sin(pi*x[1])")
>
> # 5. Apply Dirichlet BC
> apply_boundary_condition(value=0, boundary="True")  # u=0 everywhere on ∂Ω
>
> # 6. Assemble and explain each step
> ```

Then run assembly as above and explain the output.

## Expected Outputs

### Matrix Spy Pattern

For a 4×4 triangular mesh (25 DOFs, 9-point stencil):
```
Stiffness matrix sparsity:
  dims: [25, 25]
  nnz: 133
  avg_per_row: 133/25 = 5.3 nnz

Typical pattern (node 12 interior):
  K[12, 11] K[12, 12] K[12, 13]
  K[12,  7]           K[12, 17]
  K[12,  6] K[12,  7] K[12,  8]
  (and diagonal/off-diagonals for 9-point stencil)
```

### Form Rank Validation

If user asks why forms matter:

> **Form rank** tells DOLFINx what to assemble:
>
> - **Rank-2 form** (bilinear): `inner(grad(u), grad(v))*dx`
>   → Assembles to matrix K
>
> - **Rank-1 form** (linear): `f*v*dx`
>   → Assembles to vector F
>
> - **Rank-0 form** (functional): `inner(grad(u), grad(u))*dx`
>   → Assembles to scalar (energy, norm, error)

## Common Questions Answered

### "Why is the matrix sparse?"

> Basis functions have local support: each φᵢ is nonzero only on cells touching DOF i.
> So K[i,j] is nonzero only if φᵢ and φⱼ share a cell.
> For Lagrange P1 elements, this is a few neighboring DOFs, not all 81.

### "Can I inspect the assembled matrix?"

> The `assemble(target="matrix")` tool computes K but doesn't directly expose entries.
> However, you can inspect properties:
> - Matrix dimensions: should be (num_dofs, num_dofs)
> - Nonzero count: typical 5-10x per row for 2D unstructured meshes
>
> For fine-grained inspection, use `run_custom_code()` to access DOLFINx's matrix directly.

### "What happens to the RHS when I apply a BC?"

> The "lifting" method modifies the RHS F:
> 1. Create u_D satisfying boundary condition
> 2. Compute F_lifted = F - K @ u_D
> 3. Solve K @ u_interior = F_lifted
> 4. Final solution: u = u_interior + u_D
>
> Result: constrained DOFs are eliminated, no extra unknowns in the system.

### "How much does assembly cost?"

> Typical cost on modern hardware: ~10M cells/second (P1, 1-2 second JIT on first run).
>
> For your 16-cell mesh:
> - JIT compilation: ~1-2 seconds (one-time)
> - Assembly: < 1 millisecond
> - Solve: < 1 millisecond
> - Total: dominated by JIT, not the small mesh
>
> For 1M-cell mesh:
> - JIT: ~2 seconds
> - Assembly: ~0.1 seconds
> - Solve: ~0.5 seconds (depending on iterative solver)

## Detailed Output Template

When fully explaining assembly for an existing problem:

```
═══════════════════════════════════════════════════════════════
                     ASSEMBLY WALKTHROUGH
═══════════════════════════════════════════════════════════════

PROBLEM SETUP
─────────────
Mesh:           unit_square_mesh (triangle, 8×8 = 64 cells, 81 vertices)
Function space: V (Lagrange, degree=1, 81 DOFs)
PDE:            -∇²u = f in Ω, u=0 on ∂Ω

WEAK FORM
─────────
Bilinear:   a(u,v) = ∫ ∇u·∇v dx
Linear:     L(v) = ∫ f·v dx

ASSEMBLY SIMULATION
───────────────────
Bilinear assembly → K
  assemble(target="matrix", form="inner(grad(u), grad(v))*dx", apply_bcs=True)
  Result:
    dims: [81, 81]
    nnz:  451
    nnz_per_row: 5.6 (sparse ✓)

Linear assembly → F
  assemble(target="vector", form="f*v*dx", apply_bcs=True)
  Result:
    size: 81
    norm: 2.34

Diagnostic: source energy
  assemble(target="scalar", form="f*f*dx")
  Result:
    value: 6.89 (||f||_L2 ≈ 2.62)

BOUNDARY CONDITION HANDLING
───────────────────────────
Applied: u = 0 on x[0] < 1e-14
  Constrained DOFs: 9 (left edge vertices)
  Method: Lifting
    - Interior DOFs: 72 (to be solved)
    - Boundary DOFs: 9 (known = 0)
    - K rows for boundary DOFs: zeroed, diagonal = 1
    - RHS for boundary DOFs: set to 0

SOLVE PREVIEW
─────────────
System to solve: K @ u = F
  Matrix:    81×81, 451 nonzeros
  RHS:       81 components
  Unknowns:  81 DOFs total, 9 boundary values fixed, 72 to compute

Expected solution:
  - u = 0 on ∂Ω (from BC)
  - u > 0 in interior (for positive f)
  - ||u||_L2 ≈ (depends on f and domain size)

═══════════════════════════════════════════════════════════════
```

## Integration with Other Commands

- `/solve-poisson`: Uses this explanation as pre-flight check
- `/fem-solver` agent: Calls this to validate form before solve
- `explain-assembly` + `plot_solution`: Show assembly → solve → visualization pipeline

## Troubleshooting

If explanation fails:

| Issue | Fix |
|-------|-----|
| No mesh | `get_session_state()` shows "active_mesh: null". Call `create_unit_square()` first. |
| No function space | No spaces defined. Call `create_function_space()` first. |
| No forms | No bilinear/linear. Call `define_variational_form()` first. |
| Assembly crashes | Form syntax error (missing dx, rank mismatch). Check error message. |

## Files Referenced

- **SKILL.md** (same directory): Assembly pedagogy overview
- **assembly-internals.md** (references/): Deep technical details on quadrature, Jacobians, FFCx
- **formulation-architect agent**: For help designing correct weak forms
- **solver-optimizer agent**: For help after assembly with solver tuning
