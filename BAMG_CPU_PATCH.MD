# BoomerAMG CPU Patch Notes for SPSFD

This file documents the SPSFD-specific HYPRE/BoomerAMG CPU-side patch series that was introduced while debugging large voxel-domain runs on host OpenMP/MPI backends.

Commit range in the vendored HYPRE submodule:

```text
start:   495e97cbdbf231f9c3f6f146442ea6686a4f1a59
current: e9004c6eaea821f1ba05f465d1895c9f6b07bc22
```

The range contains two HYPRE commits:

```text
b9e620df53291cfe562fb135941440a2135f9fcb  hypre poor CPU convergence debug
e9004c6eaea821f1ba05f465d1895c9f6b07bc22  Brought CPU to working condition with interp_type = 17
```

The patch is intentionally local to the vendored HYPRE tree. It is not a vanilla HYPRE change and should be treated as part of the SPSFD numerical backend.

## Why This Patch Exists

SPSFD uses HYPRE BoomerAMG as the main preconditioner backend for the distributed HYPRE solver targets. The GPU target was able to build usable AMG hierarchies for large pore-scale Stokes systems, while the host OpenMP/MPI target had two classes of failures:

1. Large host runs could spend a very long time inside U-block BoomerAMG setup or be killed by Slurm due to extreme memory growth during coarse-operator construction.
2. Even when setup completed, the CPU hierarchy could be much weaker than the GPU hierarchy for the same input problem and nominal AMG parameters.

The problematic matrix was the velocity block used by the Schur/BFBT preconditioner. The pressure Schur matrix usually behaved more predictably. Diagnostics showed that the velocity block has a positive diagonal, but it is large, nonsymmetric, weakly diagonally dominant in many rows, and topologically irregular because it comes from voxel geometry and boundary-dependent finite-difference stencils.

The important observation was that the GPU BoomerAMG setup path was not failing in the same way. The goal of this patch was therefore not to change the mathematical preconditioner into a different method, but to make the host setup path closer to the device setup behavior and to expose enough internal diagnostics to find where CPU setup diverges from GPU setup.

## Original BoomerAMG CPU Setup Pipeline

Before the patch, the host setup path followed the usual BoomerAMG sequence.

For each level `l`, given a ParCSR operator `A_l`:

1. Build a strength-of-connection matrix `S_l` from `A_l`.
2. Select coarse and fine points using the requested coarsening algorithm, for example PMIS when `coarsen_type = 8`.
3. Build interpolation `P_l` from the original operator and strength graph.
4. Build the next Galerkin coarse operator:

   ```text
   A_{l+1} = R_l A_l P_l
   ```

   In the default symmetric-restriction case used here:

   ```text
   R_l = P_l^T
   A_{l+1} = P_l^T A_l P_l
   ```

5. Stop when the maximum level or coarse-size threshold is reached, then set up the selected coarse solve.

On the original CPU PMIS path, the host code maintained both local graph vertices and off-rank graph vertices explicitly. It selected independent-set candidates using local measure thresholds, exchanged C/F markers, and updated the graph. This is standard for HYPRE host coarsening, but for the SPSFD velocity block the resulting hierarchy could become much denser or weaker than the GPU hierarchy.

The relevant original host path was:

```text
hypre_BoomerAMGSetup
  hypre_BoomerAMGCreateS / hypre_BoomerAMGCreateSabs
  hypre_BoomerAMGCoarsenPMIS
    hypre_BoomerAMGCoarsenPMISHost
  hypre_BoomerAMGBuild*Interp
  hypre_BoomerAMGBuildCoarseOperatorKT or hypre_ParCSRMatrixRAPKT
```

## Main Mathematical Change: Device-Like Host PMIS

The main production change is a modified host PMIS mode controlled by:

```text
HYPRE_PMIS_HOST_DEVICE_LIKE=1
```

The older alias is also accepted:

```text
HYPRE_BAMG_PMIS_HOST_DEVICE_LIKE=1
```

The SPSFD launchers enable the new option by default for HYPRE runs unless explicitly disabled for diagnostics.

### What PMIS Is Doing

For the strength graph `G_l = (V_l, E_l)`, PMIS constructs a maximal independent set of coarse points. Each active vertex has a measure `m_i`, roughly the number of points influenced by `i`, plus a random tie-break component. The algorithm repeatedly:

1. Selects a candidate independent set from active vertices.
2. Resolves conflicts between strongly connected candidates using the measures.
3. Marks selected candidates as C-points.
4. Marks points influenced by those C-points as F-points.
5. Removes decided C/F points from the active graph.

The numerical quality of the final interpolation and coarse operator is sensitive to this C/F split. For the SPSFD velocity block, small differences in host-vs-device PMIS behavior were enough to produce very different hierarchies.

### What Changed In Host PMIS

The patched host PMIS mode tries to follow the device decision order more closely while staying inside the host ParCSR data structures.

The important changes are:

- Off-rank graph vertices are not kept as a normal active graph in device-like mode. Remote state is used through communication instead of being evolved as a separate host graph list.
- Local points with `measure < 1` are finalized as F-points during graph initialization in device-like mode.
- Active local graph vertices are admitted into the independent-set candidate stage in the device-like path, instead of applying the original host `measure > 1` candidate guard at that point.
- Conflict removal still uses measure comparisons on strong neighbors.
- Boundary conflicts use a temporary external marker instead of immediately forcing local C-points to zero in cases where the point has already been finalized.
- After every C/F update, the host device-like path synchronizes measures across ranks. This mirrors the device behavior more closely: once a point is finalized as C/F, its measure becomes zero and it must no longer suppress candidates on neighboring ranks.
- A final singleton active graph is explicitly finalized as a C-point. This prevents an infinite loop when a single point remains with a measure that does not satisfy the original host candidate condition.

In simplified notation, the patched device-like host loop is:

```text
initialize measures m_i and active set G
for i in local rows:
    if isolated(i): mark special fine/coarse as before
    elif m_i < 1: mark F and set m_i = 0
    else: keep i active

while global_active_count > 0:
    if global_active_count == 1:
        mark remaining active point C
        break

    propose active local candidates
    remove candidate conflicts using measure priority
    exchange boundary C/F conflict information
    finalize selected C-points and adjacent F-points
    exchange C/F markers
    exchange updated measures
    rebuild active graph from undecided local points
```

This does not change the final Galerkin formula. It changes the host C/F selection so that the resulting hierarchy is closer to the GPU hierarchy.

## Secondary Mathematical/Algorithmic Options

Several options were added or exposed during debugging. Not all are part of the recommended production path.

### `sabs`

Public setter added:

```c
HYPRE_BoomerAMGSetSabs(HYPRE_Solver solver, HYPRE_Int Sabs)
```

When enabled, the strength matrix is built with absolute values through `hypre_BoomerAMGCreateSabs`, i.e. strength uses `abs(A_ij)` style information rather than the usual sign-sensitive strength relation.

This was useful diagnostically for nonsymmetric/mixed-sign operators, but tests showed poor convergence for SPSFD production U-block setup. Current SPSFD configs keep:

```json
"sabs": 0
```

### `coarsen_symmetric_strength`

Public setter added:

```c
HYPRE_BoomerAMGSetCoarsenSymmetricStrength(HYPRE_Solver solver, HYPRE_Int value)
```

When enabled on host memory, the setup code builds:

```text
S_coarsen = S union S^T
```

and uses `S_coarsen` only for C/F coarsening. Interpolation and Galerkin products still use the original `S` and `A`.

This tested the hypothesis that the host coarsener was too sensitive to the nonsymmetric strength graph. It did not produce the desired production behavior and can be expensive for very large U-block setup. Current SPSFD configs keep:

```json
"coarsen_symmetric_strength": 0
```

### `interp_type = 17`

The current SPSFD HYPRE configs use:

```json
"interp_type": 17
```

This is HYPRE's modified extended+i interpolation path. The important result from testing is that the patched host PMIS path plus `interp_type = 17` is the best current CPU setup among the tried options. GPU tests also indicated that `interp_type = 17` is acceptable across the tested large domains, although `interp_type = 6` remains a useful comparison baseline.

## File-Level Patch Map

### `src/parcsr_ls/par_amg_setup.c`

This is the main BoomerAMG setup driver. Changes include:

- Added setup debug logging controlled by:

  ```text
  HYPRE_BAMG_SETUP_DEBUG
  ```

  `1` prints from rank 0 only. Values above `1` print from all ranks.

- Added hierarchy fingerprint logging controlled by:

  ```text
  HYPRE_BAMG_FINGERPRINT
  ```

  This prints level-by-level matrix sizes and C/F counts for comparing host and device hierarchies.

- Added calls that fingerprint:

  ```text
  A[0]
  S
  S_coarsen
  CF markers
  P
  A_H
  ```

- Added support for `Sabs` and `coarsen_symmetric_strength` in the normal setup path.
- Added `S_for_coarsening = S_coarsen ? S_coarsen : S`, so the symmetric coarsening graph can be used without changing interpolation or RAP.
- Added detailed phase timing for:

  ```text
  strength matrix
  coarsening
  interpolation
  RAP/coarse operator
  coarse solve setup
  ```

- Fixed aggressive-coarsening bookkeeping by resetting `coarse_size` from the actual coarse matrix after `A_H` is built:

  ```text
  coarse_size = global_rows(A_H)
  ```

  This was needed because aggressive coarsening can leave `coarse_size` unset or stale before the stopping logic. In large tests this manifested as nonsensical coarse sizes in debug output and bad stopping/coarsening decisions.

### `src/parcsr_ls/par_coarsen.c`

This is where the host PMIS behavior was changed.

Changes include:

- Added the runtime switch:

  ```text
  HYPRE_PMIS_HOST_DEVICE_LIKE
  HYPRE_BAMG_PMIS_HOST_DEVICE_LIKE
  ```

- Added phase and iteration debug output for Ruge, HMIS, PMIS, and dispatch logic.
- Added device-like PMIS behavior in `hypre_BoomerAMGCoarsenPMISHost`.
- Added final singleton handling to prevent a one-point active graph from looping forever.
- Added measure synchronization after C/F assignment in the device-like PMIS path.
- Added safer boundary conflict logic so finalized points are not reset to undecided zero markers. This was important because undecided markers after PMIS can crash interpolation.

### `src/parcsr_ls/par_rap.c`

This file contains the older host coarse-operator builder. The patch added detailed debug timing for every major host RAP stage:

```text
RT comm package create
RT transpose
Pext extract/split/map
RAP_int exterior build/exchange
RAP offd map build
RAP sizing pass
RAP allocation
RAP fill pass
RAP offd map prune
RAP ParCSR create
```

This instrumentation was needed to identify whether large CPU setup was hanging in strength, coarsening, interpolation, RAP, or coarse solve setup.

### `src/parcsr_mv/par_csr_triplemat.c`

This file contains the modular ParCSR matrix-matrix and triple-matrix host path, including:

```text
hypre_ParCSRMatMatHost
hypre_ParCSRTMatMatKTHost
hypre_ParCSRMatrixRAPKTHost
```

The patch added detailed debug timing and CSR-size reporting around local/external products, off-diagonal remapping, partial adds, exchanges, and final ParCSR construction. This was used to compare host and device RAP behavior.

The host product path is also now safer around local/external partial products and debug-visible communication-map construction. This matters for `rap2`, modular RAP, and large host matrix products where off-rank columns dominate memory and setup time.

### `src/parcsr_mv/par_csr_triplemat_device.c`

Device-side RAPKT debug output was added to make GPU-vs-CPU comparisons possible at the same level of detail. The GPU algorithm itself was not intended to be changed for production behavior; the device debug output was used as the reference when tuning the host path.

### `src/seq_mv/csr_matop.c`

`hypre_CSRMatrixAddHost` was made safer for host OpenMP use:

- If the result has zero nonzero rows, return a valid empty CSR matrix immediately.
- Limit the number of OpenMP threads in the add kernel to the number of nonzero rows.

This fixed crashes/corruption seen after some host U-setup experiments and avoids launching more add threads than there is row work.

### `src/parcsr_ls/par_amgdd_setup.c`

AMGDD setup received `HYPRE_BAMG_SETUP_DEBUG` timing output. This is diagnostic only, but it keeps AMGDD setup visible when the preconditioner type is BoomerAMGDD.

### Public Header/API Touches

The patch added public and internal declarations for:

```c
HYPRE_BoomerAMGSetSabs
HYPRE_BoomerAMGSetCoarsenSymmetricStrength
hypre_BoomerAMGSetSabs
hypre_BoomerAMGSetCoarsenSymmetricStrength
hypre_BoomerAMGCoarsenPMISHost
```

The SPSFD wrapper exposes the new JSON fields:

```json
"sabs": 0,
"coarsen_symmetric_strength": 0
```

## Runtime Controls Added By The Patch

### Production switch

```bash
export HYPRE_PMIS_HOST_DEVICE_LIKE=1
```

This enables the host PMIS path that follows the GPU PMIS decision order more closely. SPSFD launchers enable it by default for HYPRE unless `--no-hypre-pmis-host-device-like` is passed.

### Setup phase debug

```bash
export HYPRE_BAMG_SETUP_DEBUG=1
```

Recommended first level. Rank 0 prints setup phase timings. Use `2` or higher only for detailed multi-rank debugging because logs become large.

### Hierarchy fingerprint

```bash
export HYPRE_BAMG_FINGERPRINT=1
```

Prints matrix and C/F hierarchy summaries. This is useful when comparing CPU and GPU hierarchy quality.

## Current Recommended SPSFD State

For the current HYPRE backend, the recommended default direction is:

```json
"coarsen_type": 8,
"interp_type": 17,
"sabs": 0,
"coarsen_symmetric_strength": 0
```

and for host HYPRE runs:

```bash
HYPRE_PMIS_HOST_DEVICE_LIKE=1
```

This is the configuration that made the large CPU OpenMP/MPI path usable in current tests.

Important practical notes:

- CPU HYPRE setup is still significantly more memory-sensitive than GPU setup.
- CPU HYPRE should normally use enough MPI ranks to keep local matrices moderate. A few MPI ranks with very many OpenMP threads can make local ParCSR matrices too large and can amplify coarse-operator memory growth.
- The patch improves host hierarchy construction, but it does not make CPU setup identical to GPU setup. It brings the host PMIS decision process closer to the GPU path.
- The default full-periodic HYPRE behavior still relies on SPSFD-side pressure-gauge regularization and relaxed stagnation handling. That logic is outside this HYPRE patch.

## What Was Tried But Not Chosen As Default

The following options were useful for diagnosis, but are not the recommended production state:

- `sabs = 1`: reduced some nonsymmetric strength sensitivity, but produced poor convergence for SPSFD U-block preconditioning.
- `coarsen_symmetric_strength = 1`: tested `S union S^T` for coarsening only, but did not solve the large-case CPU issue and can be expensive.
- `coarsen_type = 10` HMIS: could sometimes pass setup but gave poor convergence or heavy setup for the largest cases.
- Aggressive coarsening and non-Galerkin dropping: useful setup-cost controls, but not enough by themselves to recover GPU-like convergence on the problematic U-block.
- Alternative smoothers: tested variants such as host Gauss-Seidel-like relaxations, but they were not sufficient and could be too slow or insufficiently parallel for the target workloads.
- Matrix canonicalization before HYPRE setup: useful to verify CSR handoff quality, but did not explain the CPU/GPU hierarchy gap.

## How To Compare CPU And GPU Hierarchies

Use the same solver JSON and enable the HYPRE diagnostics. For a host run:

```bash
export HYPRE_PMIS_HOST_DEVICE_LIKE=1
export HYPRE_BAMG_SETUP_DEBUG=1
export HYPRE_BAMG_FINGERPRINT=1
```

For full rank-by-rank detail:

```bash
export HYPRE_BAMG_SETUP_DEBUG=2
export HYPRE_BAMG_FINGERPRINT=2
```

Then compare for each level:

```text
A_l global rows and nnz
S_l global rows and nnz
C/F counts
P_l row/column counts and nnz
A_{l+1} rows and nnz
setup timings for strength, coarsening, interpolation, RAP
```

A healthy CPU hierarchy should have level counts, operator complexity, and coarse-grid decay close enough to the GPU hierarchy that GMRES convergence is not dominated by U-block preconditioner degradation.

## Known Limitations

- This is a vendored HYPRE patch, not an upstream HYPRE feature.
- The host PMIS path is made more GPU-like, but not bitwise or algorithmically identical to the device implementation.
- Large host runs can still need substantial RAM. This patch improves setup behavior, but it cannot remove the memory cost of building large ParCSR hierarchies.
- The patch was developed against SPSFD's voxel Stokes operators. Other nonsymmetric operators may need separate validation.
- The code currently exposes several diagnostic controls that should remain disabled in production runs unless debugging a setup/convergence issue.

## Short Summary

The patch keeps the same high-level BoomerAMG method:

```text
S_l = strength(A_l)
C/F = PMIS(S_l)
P_l = interpolation(A_l, S_l, C/F)
A_{l+1} = P_l^T A_l P_l
```

The key production change is that CPU PMIS can now use a device-like decision path. This avoids the host-only pathological C/F behavior observed on very large SPSFD velocity blocks and makes CPU OpenMP/MPI HYPRE runs viable for data sets that cannot fit into GPU memory.
