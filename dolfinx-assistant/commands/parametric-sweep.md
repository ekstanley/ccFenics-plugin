---
description: Run a parameter study across multiple values
allowed-tools: Read, Write
model: sonnet
---

Run a parametric sweep over a single parameter, solving the problem at each value and collecting output metrics.

## Setup

If $ARGUMENTS is provided, parse it as the parameter specification. Otherwise, ask the user:

1. **What parameter to vary?**
   - Mesh size (N)
   - Element degree (k)
   - Material property (name + range)
   - BC value (which BC + range)
   - Load factor (scaling coefficient)

2. **What range?**
   - For mesh size: e.g., 8, 16, 32, 64
   - For continuous parameters: min, max, number of points
   - For discrete parameters: list of values

3. **What to measure?**
   - Maximum/minimum solution value
   - L2 or H1 error (needs exact solution)
   - Value at a specific point
   - Integral quantity
   - Solver time and iteration count

## Execution

For each parameter value:

1. **Reset relevant components**: Remove old mesh/space/BCs/forms as needed
2. **Create new components**: With the current parameter value
3. **Solve**: Use the same solver configuration throughout
4. **Measure**: Extract the output metric(s)
5. **Record**: Store parameter value + all metrics

Name each solution uniquely: `u_sweep_{param}_{value}`

## Results Table

Present as a clean table:

| {Parameter Name} | {Output 1} | {Output 2} | DOFs | Solver Time (s) | Converged |
|-----------------|-----------|-----------|------|----------------|-----------|

## Analysis

After the sweep:

1. **Trend**: Is the output monotonically increasing/decreasing? Linear? Nonlinear?
2. **Convergence**: For mesh/degree sweeps, compute convergence rates between consecutive points
3. **Optimal value**: Where is the best accuracy-to-cost ratio?
4. **Anomalies**: Flag any non-converged solves or unexpected jumps

## Save Results

Write the parametric data to `/workspace/parametric_sweep.json` containing:
- Parameter name and values
- All output metrics
- Solver info per run
- Timestamp and session metadata

Generate a convergence/trend plot if matplotlib is available via `run_custom_code`.
