# FLORES: Flow Linear Operators: Resolvent and Eigenvalue Stability

MIT License

Copyright (c) 2026 NUMATH https://numath.dmae.upm.es

## Introduction

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
    ├── solver/
        ├── eig_simple.py            # Global stability solver (direct & adjoint)
        ├── resolvent.py             # Resolvent operator solver (matrix-free shell)
        ├── jac_red.py               # Domain-reduction utilities
        ├── save2pval.py             # Output routines (eigenvector → .pval files)
        ├── input_output.py          # Jacobian and coordinate readers
    ├── python_env_installation      #  Scripts to install the python environment
    ├── JAC/                         # Input directory for jacobian matrices
    ├── RESULTS_eig/                 # Output directory for eigenvalue runs
    ├── RESULTS_resolvent/           # Output directory for resolvent runs
    └── README.md
```

---

## Dependencies & Installation


Automated installation scripts are provided in the `python_env_installation/` folder. They handle everything: virtual environment creation, PETSc/SLEPc compilation (inplace, no `make install`), and the installation of `mpi4py`, `petsc4py`, `slepc4py`, and all extra Python dependencies. Both scripts include checkpoint logic, so they can be safely re-run if interrupted.

### `Cesvima_UPM_Installation.sh` — HPC cluster (CESVIMA / UPM)

Designed for the CESVIMA cluster at UPM. It links against the cluster's existing MPI, OpenBLAS, ScaLAPACK, and MUMPS modules (`foss/2021a` toolchain) rather than downloading them, and applies the necessary `libgfortran` path fix for the GCCcore 7.2.0 runtime.

Before running, load the required modules:

```bash
module load foss/2021a
module load MUMPS/5.4.0-foss-2021a-metis
```

Then, from the root of the repository:

```bash
cd python_env_installation
bash Cesvima_UPM_Installation.sh
```

> **Important:** Submit this script as a SLURM job — do not run it on the login node. PETSc/SLEPc compilation and the `petsc4py`/`slepc4py` builds are memory-intensive and will be killed by the login node's OOM policy.

### `Ubuntu_Installation.sh` — Local Ubuntu workstation

Designed for a standard Ubuntu desktop or laptop. It downloads and compiles all dependencies from scratch (MPICH, BLAS/LAPACK, ScaLAPACK, MUMPS, CMake, METIS, ParMETIS), so no pre-installed MPI or system libraries are required beyond a working C/Fortran compiler.

From the root of the repository:

```bash
cd python_env_installation
bash Ubuntu_Installation.sh
```

### After installation (both platforms)

Both scripts patch the virtual environment's `activate` script with the correct `PETSC_DIR`, `SLEPC_DIR`, `PETSC_ARCH`, and `LD_LIBRARY_PATH` variables. To activate the environment in future sessions:

```bash
source myvenv/bin/activate
```

A sanity check is run automatically at the end of each script, printing the PETSc and SLEPc versions and verifying MPI communication.

---

## Usage

### Global stability analysis (`EIGENVAL.py`)

Edit the parameter block at the top of `EIGENVAL.py`:

```python
jacfile   = 'JAC/samg.matrix.amg.pval'
volfile   = 'JAC/samg.matrix.vol'
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
jacfile    = 'JAC/samg.matrix.amg.pval'
listomegas = [57.2j, 60j, 80j]   # list of frequencies ω to sweep
nev        = 30                   # number of singular values requested
```

Submit via SLURM:

```bash
mpirun -np 8 python RESOLVANT.py
```

Optimal forcing/response modes are written to `RESULTS_resolvent/` for each frequency.

---



## Authors & Acknowledgements

Copyright (c) 2026 NUMATH https://numath.dmae.upm.es

**Development:** Alejandro Martinez-Cava, Iván Padilla, Miguel Chávez-Modena, 

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
