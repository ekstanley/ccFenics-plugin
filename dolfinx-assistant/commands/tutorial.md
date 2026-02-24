---
description: Walk through a DOLFINx tutorial chapter interactively
allowed-tools: Read, Grep, Glob
argument-hint: [chapter-name]
model: sonnet
---

Guide the user through a DOLFINx tutorial chapter step by step. Each step should explain what's happening and why, then execute it using MCP tools.

## Available Chapters

If $ARGUMENTS is empty, present these options and ask the user to pick:

| Chapter | Topic | Difficulty |
|---------|-------|-----------|
| poisson | Poisson equation on unit square | Beginner |
| elasticity | Linear elasticity | Beginner |
| heat-equation | Time-dependent heat diffusion | Intermediate |
| nonlinear-poisson | Nonlinear Poisson with Newton | Intermediate |
| stokes | Stokes flow (Taylor-Hood) | Intermediate |
| mixed-bcs | Mixed boundary conditions | Intermediate |
| helmholtz | Helmholtz equation | Advanced |
| hyperelasticity | Large deformation elasticity | Advanced |
| navier-stokes | Incompressible Navier-Stokes | Advanced |
| eigenvalue | Eigenvalue problems with SLEPc | Advanced |

## Teaching Approach

For each step in the tutorial:

1. **Explain** what we're about to do and why (2-3 sentences, plain language)
2. **Execute** the MCP tool call
3. **Interpret** the result — what does the output mean?
4. **Connect** to FEM theory where relevant (brief, not a lecture)

## Pacing

After each major section (mesh, space, BCs, solve, results), pause and ask:
- "Any questions about this step?"
- "Ready to continue?"

## After Completion

1. Summarize what was built and the key concepts covered
2. Suggest a modification the user could try (different BCs, finer mesh, higher degree)
3. Recommend the next tutorial chapter based on difficulty progression

If $ARGUMENTS is provided, map it to the appropriate chapter and begin immediately.
