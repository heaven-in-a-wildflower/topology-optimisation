# Topology Optimisation for Thermal and Elastic Problems

For details, see the [report](https://drive.google.com/file/d/1GZPFteMixybQ7XR65yN_OSpCgBEsfkE-/view?usp=sharing) and [presentation](https://www.overleaf.com/read/nzvrsswwptqq#7c995a).

This is exploratory C++ code built on the [MFEM](https://mfem.org) finite-element library. It covers:

- Finite-element heat-conduction solvers, steady-state and transient, for a 3D mesh. They run in parallel with MPI and hypre.
- Thermal topology optimisation in 2D with the SIMP density method, steady-state and transient.
- A level-set compliance-minimisation experiment for a 2D cantilever.

`topopt.py` is the classic 165-line Python SIMP code by Niels Aage and Villads Egede Johansen (2013), kept for reference.

This README describes the `master` branch. The `submission` branch is a separate snapshot from November 2025, prepared for the B.Tech project submission. It adds a 3D SIMP solver (`simp_3d.cpp`), the shell scripts `pipeline.sh` and `all_steady.sh`, `heat_stats.py` and thermal-analysis plots. It drops the level-set code and `topopt.py`.

## Contents

| Path | Description |
| --- | --- |
| `heat_cond_steady_state.cpp` | Steady, nonlinear heat conduction on a 3D mesh using Picard iteration (MPI-parallel) |
| `heat_cond_transient.cpp` | Transient heat conduction with backward or forward Euler time stepping (MPI-parallel) |
| `simp.cpp` | Steady-state thermal topology optimisation with SIMP on a 2D square |
| `simp_transient.cpp` | Transient thermal topology optimisation with SIMP on a 2D square |
| `levelset_cantilever.cpp` | Level-set compliance minimisation for a 2D cantilever. The commit that added it calls it "non-functional" |
| `topopt.py` | Aage & Johansen's 165-line Python SIMP code for minimum compliance (reference) |
| `heat_cond_steady_state`, `heat_cond_transient`, `simp`, `simp_transient`, `levelset_cantilever` | Prebuilt Linux x86-64 binaries of the programs above. They were built before later fixes to the source, so some printed messages are outdated |
| `heat_cond_steady_state_solns/`, `heat_cond_transient_solns/` | Saved meshes and temperature fields from earlier heat-conduction runs |
| `simp_steady_state_solns/`, `simp_transient_solns/` | Saved meshes and final densities (continuous and thresholded) from earlier SIMP runs |
| `levelset_results/` | Saved mesh, density, level-set and displacement fields, plus `history.txt` (compliance and volume per iteration) |
| `step_files/cylinder.step` | CAD geometry of the cylinder used for the heat-conduction meshes |
| `report.txt` | Outline of the project report |
| `dump.txt` | Another version of the level-set code. It adds an adaptive Lagrange multiplier and a normalised velocity, following Allaire et al. (2004) |
| `logs.txt` | Console log of a 100-iteration run of the `dump.txt` version |
| `level_set_removed.pdf` | Allaire, Jouve & Toader, "Structural optimization using sensitivity analysis and a level-set method", *J. Comput. Phys.* 194 (2004) 363–393 |

The `.msh` meshes the heat solvers read are not in the repository. They belonged in `mesh_files/`, which is listed in `.gitignore`, and `.gitattributes` sets that folder up for Git LFS.

The saved results are in MFEM's mesh and grid-function format. The `.NNNNNN` suffix on the heat-conduction outputs is the MPI rank; the saved runs used a single process.

## Programs

### Heat conduction (`heat_cond_*.cpp`)

These use first-order H1 elements, distributed across MPI ranks with `ParMesh`. Each linear system is solved with hypre's preconditioned conjugate-gradient solver (PCG) and the BoomerAMG preconditioner.

**`heat_cond_steady_state.cpp`**

- Solves −∇·(k(T)∇T) = 0 with a temperature-dependent conductivity k = κ + αT. By default κ = 205 and α = 0.01.
- Handles the nonlinearity with Picard iteration: at most 100 iterations, stopping when the relative change falls below 10⁻⁶.
- Boundary conditions, by mesh boundary attribute. Attributes the mesh doesn't have are skipped:

  | Attribute | Condition | Default |
  | --- | --- | --- |
  | 1 | Heat flux | q = 1000 |
  | 2, 3 | Convection | h = 20, T_amb = 25 |
  | 4, 5 | Fixed temperature | T = 100 |

- Main options: `-m` (mesh), `-o` (order), `-k`, `-a`, `-q`, `-ta`, `-mi` (maximum Picard iterations) and `-tol`.
- Writes its output to `heat_cond_steady_state_solns/`.

**`heat_cond_transient.cpp`**

- Solves ρc_p ∂T/∂t = ∇·(κ∇T).
- Boundary conditions: heat flux q = 10⁶ on attribute 1, and convection (h = 20, T_amb = 10) on attribute 2. There are no fixed-temperature boundaries.
- Material values: ρ = 2700 and c_p = 900, as for aluminium, but κ = 10⁶.
- Starts from a uniform temperature equal to T_amb. Smooth quadratic and cosine starting profiles are in the code but commented out.
- Time stepping:
  - The default (`-imp`) is backward Euler: (M + Δt K) Tⁿ⁺¹ = M Tⁿ + Δt f. The PCG/AMG solver is set up once and reused while Δt stays the same.
  - `-no-imp` switches to forward Euler with a lumped mass matrix. The run stops with "Instability detected!" if |∂T/∂t| exceeds 10⁶.
- Main options: `-m` (mesh), `-o` (order), `-tf` (final time, default 500), `-dt` (time step, default 0.001), `-imp`/`-no-imp`, and `-nr` (print timing breakdowns every n steps).
- Writes its output to `heat_cond_transient_solns/`.

### SIMP thermal topology optimisation (`simp.cpp`, `simp_transient.cpp`)

Both programs run in a single process, and all their parameters are compile-time constants in `main`.

**Common setup**

- **Mesh:** a unit square of 80 × 80 bilinear quadrilaterals.
- **Design variable:** a density ρ, starting uniform at the target volume fraction of 0.3.
- **Material model:** conductivity k = k_min + (1 − k_min) ρᵖ, with k_min = 10⁻⁹.
- **Filter:** a PDE (Helmholtz) density filter with r_min = 0.02, applied to both densities and sensitivities.
- **Update rule:** optimality criteria with a move limit of 0.05. The Lagrange multiplier is found by bisection to meet the volume constraint.
- **Penalty schedule:** the exponent p stays at 1 for the first 10 iterations, then grows by ×1.5 every 5 iterations, up to 8.
- **Stopping:** at most 150 iterations; after iteration 15, stop when the relative change in compliance is below 10⁻⁵.
- **Output:** the final filtered density, plus a binary design thresholded at ρ > 0.5.

**`simp.cpp` (steady state)**

- A uniform heat source of 1 over the whole domain.
- Heat leaves only through a small convective strip in the middle of the left edge (x < 0.01, 0.45 < y < 0.55), with h = 50 and T_amb = 25.
- The objective is thermal compliance q·T.
- A constant `USE_CONVECTION_BC = false` switches to a fixed temperature on the left edge instead.
- Results go to `simp_steady_state_solns/`.

**`simp_transient.cpp` (transient)**

- A heat source of 10 along the left edge (x < 0.01), switched on for the first 10 s.
- Convection (h = 10, T_amb = 25) on the right edge (x > 0.99).
- Heat capacity is 2700 · 900 · ρ.
- Each design iteration runs backward Euler with Δt = 0.05 up to t = 20 s, which is 400 steps.
- The objective is the time-integrated compliance Σ (q·T) Δt.
- Sensitivities are summed over the stored temperature history. This is a simplified direct method, not a full adjoint.
- Results go to `simp_transient_solns/`.

### Level-set cantilever (`levelset_cantilever.cpp`)

- **Problem:** minimise compliance of a 2 × 1 plane-elasticity domain on a 160 × 80 grid, with E = 1, ν = 0.3 and an "ersatz" (very weak) void material with E = 10⁻³.
- **Supports and load:** boundary attribute 1 is fixed, and a unit downward point load acts at the middle of the right edge.
- **Starting design:** a 6 × 3 grid of holes.
- **Each iteration:**
  1. Solve the elasticity problem.
  2. Compute a shape velocity from the strain energy minus a fixed Lagrange multiplier of 5.
  3. Advance a Hamilton–Jacobi equation for the level-set function over 10 upwind steps.
  4. Every 5 iterations, reinitialise the level-set function.
- **Stopping:** after iteration 10, stop when the compliance changes by less than 0.5% over 5 iterations.
- **Output:** results go to `levelset_results/`.

The saved run stopped after 12 iterations, with compliance falling from about 2.51 × 10⁶ to 2.34 × 10⁶ at a volume fraction of about 0.40.

## Building

There is no build system. The comments at the top of each `.cpp` file give the compile command. They assume MFEM 4.8, hypre, METIS 4.0, OpenBLAS and GLVis 4.4 sit in folders next to this repository:

```bash
mpicxx -O3 -std=c++17 -fopenmp -I../mfem-4.8/ -I../hypre/src/hypre/include simp.cpp -o simp \
    -L../mfem-4.8 -lmfem -L../hypre/src/hypre/lib -lHYPRE -L../metis-4.0 -lmetis -lrt -lopenblas
```

Replace `simp` with `heat_cond_steady_state`, `heat_cond_transient`, `simp_transient` or `levelset_cantilever` as needed. The build comment in `levelset_cantilever.cpp` was added later, following the other files, and has not been tested.

The committed binaries link MFEM and hypre statically. `readelf` shows they only need Open MPI's `libmpi.so.40` and `libmpi_cxx.so.40` plus the standard C/C++ runtime.

## Running

```bash
# Heat conduction; add `mpirun -np <N>` to run in parallel
./heat_cond_steady_state -m mesh_files/two_holes_ref1.msh
./heat_cond_transient -m mesh_files/through_hole.msh -tf 100 -dt 0.01 -nr 10

# Topology optimisation (no command-line options)
./simp
./simp_transient
./levelset_cantilever

# Visualise saved results with GLVis
glvis -m heat_cond_steady_state_solns/cylinder_mfem_mesh.000000 -g heat_cond_steady_state_solns/cylinder_steady_solution.000000
glvis -m heat_cond_transient_solns/cylinder_mfem_mesh.000000 -g heat_cond_transient_solns/cylinder_transient_solution.000000
glvis -m simp_steady_state_solns/steady_thermal.mesh -g simp_steady_state_solns/steady_thermal_final.gf
glvis -m simp_transient_solns/transient_thermal.mesh -g simp_transient_solns/transient_thermal_final.gf
glvis -m levelset_results/cantilever.mesh -g levelset_results/density_final.gf

# Python reference code (needs numpy, scipy and matplotlib)
python3 topopt.py [nelx nely volfrac rmin penal ft]   # defaults: 180 60 0.4 5.4 3.0 1
```

## Notes and known issues

- The heat solvers need `.msh` meshes that are not in the repo; the default mesh name is `cylinder_3d.msh`.
- A mesh generated from `step_files/cylinder.step` must use the boundary-attribute numbering listed above.
- In `logs.txt`, the `dump.txt` level-set variant ran all 100 iterations with the volume fraction stuck at about 0.338, against its target of 0.4.
