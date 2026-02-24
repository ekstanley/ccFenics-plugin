---
description: Guided simulation setup — mesh, elements, BCs, materials
allowed-tools: Read, Grep, Glob
---

Walk the user through setting up a DOLFINx simulation step by step. Use AskUserQuestion at each stage to gather inputs. Do NOT proceed to solving — only configure.

## Step 1: Problem Definition

Ask the user:
- What PDE are they solving? (Poisson, elasticity, Stokes, heat equation, custom)
- What is the domain geometry? (unit square, rectangle, box, custom mesh file)
- Spatial dimension? (2D or 3D)

## Step 2: Mesh Creation

Based on the domain:
- For simple domains: use `create_mesh` with appropriate parameters
- For custom geometry: use `create_custom_mesh` with a .msh file
- Ask about mesh resolution (coarse for testing, fine for production)
- Run `compute_mesh_quality` and report results

## Step 3: Element Selection

Reference the element-selection skill. Ask:
- Is the problem scalar or vector?
- Any special requirements (incompressibility, curl-conforming)?
- Accuracy vs cost preference?

Create the function space with `create_function_space`.

## Step 4: Material Properties

Ask about coefficients:
- Constant values or spatially varying?
- For each coefficient, get the value or expression

Define with `set_material_properties`.

## Step 5: Boundary Conditions

Ask about each boundary:
- Which boundaries have Dirichlet BCs? What values?
- Any Neumann (natural) BCs?
- Any Robin BCs?

Use `mark_boundaries` if needed, then `apply_boundary_condition` for each.

## Step 6: Summary

Present a complete summary table:
- PDE type, domain, mesh size
- Element family and degree
- Material parameters
- Boundary conditions
- Recommended solver (reference solver-selection skill)

Confirm everything looks correct before the user moves to solving.
