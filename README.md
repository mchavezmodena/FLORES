# FLORES

MIT License

Copyright (c) 2021 NUMATH https://numath.dmae.upm.es


## Flow Linear Operators: Resolvent and Eigenvalue Stability

**FLORES** is a parallel Python toolkit for global stability analysis and resolvent-based input–output analysis of compressible flows. It computes eigenvalues, eigenmodes, and optimal forcing/response modes using PETSc/SLEPc and the MUMPS direct solver, and is designed to run on HPC clusters via MPI.

---

## Table of Contents

- [Background](#background)
- [Mathematical Formulation](#mathematical-formulation)
- [Repository Structure](#repository-structure)
- [Dependencies & Installation](#dependencies--installation)
- [Usage](#usage)
- [Test Case: Backward-Facing Step](#test-case-backward-facing-step)
- [Authors & Acknowledgements](#authors--acknowledgements)
- [License](#license)

---

## Background

FLORES targets two classical problems in hydrodynamic stability theory:

1. **Global eigenvalue analysis** — identifies the natural oscillatory modes of a flow (oscillator behaviour).
2. **Resolvent analysis** — quantifies the linear amplification of external forcing by computing the leading singular values and modes of the resolvent operator (amplifier behaviour).

Both analyses are performed on a linearised Navier–Stokes Jacobian produced by an external CFD solver.

---

## Mathematical Formulation

### Global Stability (eigenvalue problem)

The linearised flow dynamics are governed by

```
dq'/dt = A q'
```

where `A` is the Jacobian of the nonlinear residual. The generalised eigenvalue problem

```
A x = λ B x
```

is solved with a shift-and-invert spectral transformation centred at a user-defined target `σ` in the complex plane. The mass matrix `B` contains the cell volumes. Direct and adjoint problems are both supported.

### Resolvent Analysis

For a temporal frequency `ω`, the resolvent operator is defined as

```
C(ω) = Pᵀ (iω B − A)⁻¹ P
```

where `P` is a prolongation matrix that restricts the input/output space to the momentum equations. The leading singular triplets of `C(ω)` are obtained by recasting the SVD as an eigenvalue problem

```
[C(ω) C(ω)†] f̂ = σ² f̂
```

which is solved with SLEPc. A single LU factorisation of `(iω B − A)` via MUMPS is reused for both the forward (`ksp.solve`) and adjoint (`ksp.solveTranspose`) solves, making the computation efficient for a sweep over multiple frequencies.

---

## Repository Structure

```
FLORES/
├── EIGENVAL.py          # Global stability solver (direct & adjoint)
├── RESOLVANT.py         # Resolvent operator solver (matrix-free shell)
├── jac_red.py           # Domain-reduction utilities
├── save2pval.py         # Output routines (eigenvector → .pval files)
├── input_output.py      # Jacobian and coordinate readers
├── RESULTS_eig/         # Output directory for eigenvalue runs
├── RESULTS_resolvent/   # Output directory for resolvent runs
└── README.md
```

---

## Dependencies & Installation

### Cluster module stack (tested configuration)

```bash
module load foss/2021a                       # OpenMPI 4.1.1, OpenBLAS, ScaLAPACK
module load MUMPS/5.4.0-foss-2021a-metis
```

### PETSc / SLEPc (version 3.25.0, inplace build)

```bash
# Configure — use the cluster MPI, not downloaded MPICH
./configure \
    --with-mpi-dir=$EBROOTOPENMPI \
    --download-cmake \
    --download-metis \
    --download-parmetis \
    --with-scalar-type=complex \
    --with-precision=double \
    PETSC_ARCH=arch-complex

make PETSC_DIR=$(pwd) PETSC_ARCH=arch-complex all
```

> **Note:** Do not run `make install`. FLORES uses the inplace build directory.

### Python packages

```bash
pip install mpi4py --break-system-packages

# petsc4py and slepc4py must exactly match the compiled PETSc/SLEPc version
CFLAGS="-O0 -g0" MAX_JOBS=1 pip install petsc4py==3.25.0 --break-system-packages
CFLAGS="-O0 -g0" MAX_JOBS=1 pip install slepc4py==3.25.0 --break-system-packages

pip install numpy scipy netCDF4 --break-system-packages
```

> **Important:** Compile petsc4py/slepc4py as a SLURM job, not on the login node, to avoid out-of-memory kills.

### Environment variables

```bash
export PETSC_DIR=/path/to/petsc
export PETSC_ARCH=arch-complex
export SLEPC_DIR=/path/to/slepc
```

---

## Usage

### Global stability analysis (`EIGENVAL.py`)

Edit the parameter block at the top of `EIGENVAL.py`:

```python
jacfile   = 'path/to/samg.matrix.amg.pval'
volfile   = 'path/to/samg.matrix.vol'
nev       = 50          # number of eigenvalues requested
the_shift = -0.05 + 4j  # shift in the complex plane
adjoint   = False       # set True for adjoint problem
```

Submit via SLURM:

```bash
mpirun -np 8 python EIGENVAL.py
```

Converged eigenvalues are appended to `RESULTS_eig/eigv_DIR.dat` (or `eigv_ADJ.dat`). Eigenvectors are written as `RESULTS_eig/eigf_N.pval`. Duplicate detection across restarts is built in.

### Resolvent analysis (`RESOLVANT.py`)

Edit the parameter block:

```python
jacfile    = 'path/to/samg.matrix.amg.pval'
listomegas = [57.2j, 60j, 80j]   # list of frequencies ω to sweep
nev        = 30                   # number of singular values requested
```

Submit via SLURM:

```bash
mpirun -np 8 python RESOLVANT.py
```

Optimal forcing/response modes are written to `RESULTS_resolvent/` for each frequency.

---

## Test Case: Backward-Facing Step

The default paths point to a backward-facing step (BFS) configuration located at

```
/projects/07_TRANSDIFFUSE/02_BFS/JAC/
```

This case uses:
- Mach number `M = 0.1`
- Spanwise wavenumber `β = 0`
- Target shift `σ = −0.05 + 4i` for the eigenvalue solver
- Frequency sweep `ω = 57.2i` for the resolvent solver

---

## Authors & Acknowledgements

**Development:** Miguel Chávez, Iván Padilla,  (UPM / TRANSDIFFUSE project)

**Original implementation:** The resolvent and eigenvalue solver architecture is based on the original code developed by **Alejandro Martínez Cava** as part of his doctoral thesis at the Universidad Politécnica de Madrid (UPM). His foundational work on the matrix-free resolvent operator and the PETSc/SLEPc solver infrastructure made this tool possible. Martínez-Cava Aguilar, Alejandro  (2019). Direct and Adjoint Methods for Highly Detached Flows. Tesis (Doctoral), E.T.S. de Ingeniería Aeronáutica y del Espacio (UPM). https://doi.org/10.20868/UPM.thesis.56391. 

This work is part of the **TRANSDIFFUSE** project at UPM.

---

```
MIT License

Copyright (c) 2025 Miguel, Universidad Politécnica de Madrid

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```
