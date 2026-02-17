---
name: fem-workflow-eigenvalue
description: |
  Guides the user through solving eigenvalue problems using DOLFINx MCP tools and SLEPc.
  Use when the user asks about eigenvalue problems, eigenmodes, modal analysis,
  natural frequencies, vibration modes, waveguide modes, or spectral problems.
---

# Eigenvalue Problem Workflow (Official Demo)

Solve **generalized eigenvalue problems** A*x = lambda*B*x using SLEPc.

## Applications

- **Structural vibration**: Natural frequencies and mode shapes
- **Acoustics**: Resonant frequencies of cavities
- **Electromagnetics**: Waveguide cutoff frequencies
- **Stability analysis**: Buckling loads

## Laplacian Eigenvalues Example

Find the eigenvalues of -lap(u) = lambda*u on the unit square with homogeneous Dirichlet BCs.

### Exact Solution

On [0,1]^2: lambda_{m,n} = pi^2*(m^2 + n^2), so lambda_1 = 2*pi^2 ~ 19.739.

### Step-by-Step Tool Sequence

#### 1. Create the Mesh

```
create_unit_square(name="mesh", nx=16, ny=16)
```

#### 2. Create Function Space

```
create_function_space(name="V", family="Lagrange", degree=1)
```

#### 3. Apply Boundary Conditions

```
apply_boundary_condition(value=0.0, boundary="True")
```

#### 4. Solve Eigenvalue Problem

```
solve_eigenvalue(
    stiffness_form="inner(grad(u), grad(v))*dx",
    mass_form="inner(u, v)*dx",
    num_eigenvalues=6,
    which="smallest_magnitude",
    function_space="V"
)
```

#### 5. Inspect Results

The tool returns eigenvalues and stores eigenvectors as functions (eig_0, eig_1, ...).

```
compute_error(function_name="eig_0")
```

## EM Waveguide Example

Find cutoff frequencies of a rectangular waveguide using curl-curl operator.

### 1. Create Mesh and Nedelec Space

```
create_unit_square(name="mesh", nx=16, ny=16)
create_function_space(name="V", family="N1curl", degree=1)
```

### 2. Apply PEC Boundary Conditions

```
apply_boundary_condition(value=0.0, boundary="True", function_space="V")
```

### 3. Solve

```
solve_eigenvalue(
    stiffness_form="inner(curl(u), curl(v))*dx",
    mass_form="inner(u, v)*dx",
    num_eigenvalues=6,
    which="smallest_magnitude",
    function_space="V"
)
```

## Solver Type Selection

| Type | Best For |
|------|----------|
| `krylovschur` | General purpose, default, robust |
| `lanczos` | Symmetric problems (Hermitian matrices) |
| `arnoldi` | Non-symmetric problems |
| `lapack` | Small problems (dense solver, exact) |
| `power` | Dominant eigenvalue only |

## Which Eigenvalues

| Option | Meaning |
|--------|---------|
| `smallest_magnitude` | Smallest |lambda|| (default) |
| `largest_magnitude` | Largest |lambda|| |
| `smallest_real` | Most negative Re(lambda) |
| `largest_real` | Most positive Re(lambda) |
| `target_magnitude` | Closest to target value |
| `target_real` | Closest real part to target |

For `target_*` modes, provide the `target` parameter. This uses spectral transformation (shift-invert) for efficient computation near the target.

## Common Pitfalls

- Forgetting BCs: Without Dirichlet BCs, the Laplacian has a zero eigenvalue (rigid body mode)
- Mesh too coarse: Higher eigenvalues need finer meshes for accuracy
- Wrong element for EM: Use Nedelec (N1curl) for curl-curl, not Lagrange
- Spurious eigenvalues: May appear in EM problems; filter by checking eigenvector quality
