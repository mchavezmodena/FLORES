#! /usr/bin/env python
# This file only runs in sequential mode

import numpy as np
import sys, os
from netCDF4 import Dataset
from scipy.sparse import csr_matrix, linalg as sla
from scipy.sparse import identity

from jac_red import domain_reduction
from save2pval import mode2pval, mode2pval3D
from input_output import openjacobian, read_coordinates
import pdb

import petsc4py
import slepc4py
slepc4py.init(sys.argv)
from petsc4py import PETSc
from slepc4py import SLEPc
from mpi4py import MPI


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_previous_eigenvalues(eigv_file):
    """
    Read eigenvalues already stored in eigv_file.
    Returns a list of complex numbers and the highest index found.
    """
    eigs_prev = []
    max_index  = -1
    if not os.path.isfile(eigv_file):
        return eigs_prev, max_index
    with open(eigv_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            idx   = int(parts[0])
            re    = float(parts[1])
            im    = float(parts[2])
            eigs_prev.append(complex(re, im))
            max_index = max(max_index, idx)
    return eigs_prev, max_index


def is_duplicate(new_eig, existing_eigs, tol_real=1e-5, tol_imag=1e-5):
    """
    Return True if new_eig is already present in existing_eigs within tolerance.
    Both real and imaginary parts must be within their respective tolerances.
    """
    for e in existing_eigs:
        if abs(new_eig.real - e.real) < tol_real and \
           abs(new_eig.imag - e.imag) < tol_imag:
            return True
    return False


def next_eigvec_index(results_dir):
    """
    Scan results_dir for eigf_N.pval files and return the next available index.
    """
    max_idx = -1
    if os.path.isdir(results_dir):
        for fname in os.listdir(results_dir):
            if fname.startswith('eigf_') and fname.endswith('.pval'):
                try:
                    idx = int(fname[len('eigf_'):-len('.pval')])
                    max_idx = max(max_idx, idx)
                except ValueError:
                    pass
    return max_idx + 1


# ─────────────────────────────────────────────────────────────────────────────
# Main solver
# ─────────────────────────────────────────────────────────────────────────────

def run_slices():
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    Print = PETSc.Sys.Print

    # Path to the Jacobian and volumes files
    input_file_path = '/home/w059/w059589/projects/07_TRANSDIFFUSE/02_BFS/JAC/'

    #######################
    jacfile = input_file_path + 'samg.matrix.amg.pval'
    volfile = input_file_path + 'samg.matrix.vol'
    #######################
    beta    = 0.0
    mach    = 0.1
    rlength = 1
    #######################
    dreduced = False
    xmin =  0.12;  xmax = 0.35
    zmin = -0.05;  zmax = 0.05
    #######################
    nev       = 50
    gen       = False
    adjoint   = False
    the_shift = -0.05 + 4*1j
    #######################

    # Duplicate-detection tolerances
    dup_tol_real = 1e-5
    dup_tol_imag = 1e-5

    fac = 1. / (mach * np.sqrt(1.4))

    results_dir = './RESULTS_eig'
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)

    # ── Determine output eigenvalue file ────────────────────────────────────
    if adjoint:
        eigv_file = os.path.join(results_dir, 'eigv_ADJ.dat')
    else:
        eigv_file = os.path.join(results_dir, 'eigv_DIR.dat')

    # ── Load previously computed eigenvalues (rank 0 reads, then broadcasts) -
    if rank == 0:
        eigs_prev, _ = load_previous_eigenvalues(eigv_file)
        n_prev = len(eigs_prev)
        Print(f' Found {n_prev} previously computed eigenvalue(s) in {eigv_file}')
    else:
        eigs_prev = None
        n_prev    = None

    # Broadcast to all ranks so every process can check duplicates
    eigs_prev = comm.bcast(eigs_prev, root=0)
    n_prev    = comm.bcast(n_prev,    root=0)

    # Starting index for new eigenvector files
    file_start_idx = next_eigvec_index(results_dir)

    # ── Read Jacobian ────────────────────────────────────────────────────────
    Print('')
    Print(' Reading Jacobian')
    amatrix, neq = openjacobian(jacfile)
    amatrix.data *= fac
    nvars = amatrix.shape[0]
    Print(f' Matrix main dimension = {nvars}')
    Print(f' Number of equations   = {neq}')
    Print('')

    gridpoints = int(nvars / neq)

    # ── Mass matrix ─────────────────────────────────────────────────────────
    Print(' Reading mass matrix and generating M')
    Print('')
    one = 1. + 0j
    with open(volfile, 'r') as f:
        vols = [float(line) for line in f.readlines()]
    bmatrix = identity(nvars, dtype='c16', format='csr')
    bmatrix.data[:] = [vols[i // neq] * one for i in range(nvars)]

    # ── Domain reduction ─────────────────────────────────────────────────────
    if dreduced:
        Print('')
        Print(' Applying domain reduction')
        Print(f' XMIN/XMAX = {xmin}/{xmax}')
        Print(f' ZMIN/ZMAX = {zmin}/{zmax}')
        coord = read_coordinates(input_file_path + 'samg.matrix.coo', rlength, beta)
        dr = domain_reduction(zmin, zmax, xmin, xmax)
        dr.create_Pmatrix(coord)
        Print(f' Previous NNZ = {amatrix.nnz}')
        amatrix = dr.reduce_matrix(amatrix)
        Print(f' New NNZ      = {amatrix.nnz}')
        bmatrix = dr.reduce_matrix(bmatrix)
        n = amatrix.shape[0]
        Print(f' New leading dimension of A = {n}')
        localid = np.arange(0, gridpoints, 1, dtype='i4')
        localid = np.repeat(localid, neq)
        rgid = dr.reduce_vector(localid)
        rgid = rgid[0::neq].astype(int)
        Print('')
    else:
        rgid = None
        n    = nvars

    # ── PETSc matrices ───────────────────────────────────────────────────────
    A = PETSc.Mat()
    A.create(PETSc.COMM_WORLD)
    A.setSizes([n, n])
    A.setUp()
    A.setType('seqaij')

    Print(' Assembling PETSc matrix...')
    A.setPreallocationCSR((amatrix.indptr[:],
                           amatrix.indices[:],
                           amatrix.data[:]))
    A.assemble()

    B = PETSc.Mat()
    B.create(PETSc.COMM_WORLD)
    B.setSizes([n, n])
    B.setUp()
    B.setType('seqaij')
    RowStart, RowEnd = B.getOwnershipRange()
    for pt in range(RowStart, RowEnd):
        B[pt, pt] = bmatrix.data[pt]
    B.assemble()

    # ── Problem setup ────────────────────────────────────────────────────────
    Print('\n################################')
    if adjoint:
        A.transpose()
        Print('    SOLVING ADJOINT PROBLEM')
    else:
        Print('    SOLVING DIRECT PROBLEM')
    Print('################################\n')

    ncv = nev * 3 + 1
    mpd = ncv - 1

    E = SLEPc.EPS().create()
    if gen:
        Print('  Generalised EGVP  Ax=sMx')
        E.setOperators(A, B)
    else:
        Print('  Standard EGVP   Ax=sx')
        E.setOperators(A)
        E.setProblemType(SLEPc.EPS.ProblemType.NHEP)

    ST = E.getST()
    ST.setType('sinvert')
    ST.setShift(the_shift)
    ST.setFromOptions()

    K = ST.getKSP()
    K.setType('preonly')
    K.setFromOptions()
    pc = K.getPC()
    pc.setType('lu')
    pc.setFactorSolverType('mumps')
    opts = PETSc.Options("mat_mumps_")
    opts["icntl_7"]  = 4
    opts["icntl_6"]  = 2
    opts["icntl_4"]  = 1
    opts["icntl_11"] = 2
    opts["icntl_10"] = 4
    opts["cntl_3"]   = 1e-6

    E.setTolerances(tol=1e-8, max_it=15000)
    E.setDimensions(nev, ncv, mpd)
    E.setWhichEigenpairs(E.Which.TARGET_MAGNITUDE)
    E.setTarget(complex(the_shift))
    E.setFromOptions()

    if rank == 0:
        print('')
        print(' TYPE OF KSP:                    ', K.getType())
        print(' ST Type                       = ', ST.getType())
        print(' ST Shift                      = ', ST.getShift())
        print(' Target                        = ', E.getTarget())
        print(' EGV Solver                    = ', E.getType())
        print(' EPS Problem Type              = ', E.getProblemType())
        print(' Region of Spectrum            = ', E.getWhichEigenpairs())
        print(' Number of eigenvalues requested = ', nev)
        print(' Number of column vectors        = ', ncv)
        print(' Maximum dimension (mpd)         = ', mpd)
        print('')

    E.solve()

    # ── Post-processing ──────────────────────────────────────────────────────
    its   = E.getIterationNumber()
    tol, max_it = E.getTolerances()
    ksp_its = K.getIterationNumber()
    nconv = E.getConverged()

    if rank == 0:
        print('')
        print(' Iterations (EPS)  : ', its)
        print(' Tolerance / max_it: ', tol, ' / ', max_it)
        print(' Iterations (KSP)  : ', ksp_its)
        print(' Requested         : ', nev)
        print(' Converged         : ', nconv)

    xr, xrl = A.getVecs()
    xi, xil = A.getVecs()

    new_eigs    = []   # new eigenvalues accepted in this run
    skipped     = 0
    file_idx    = file_start_idx   # running index for eigenvector files

    if nconv > 0:
        Print("")
        Print("           k           ||Ax-kx||/||kx||   status")
        Print("----------------------------------------------------")

        for i in range(nconv):
            k     = E.getEigenpair(i, xr, xi)
            error = E.computeError(i)

            # ── Duplicate check ──────────────────────────────────────────────
            already_known = is_duplicate(k, eigs_prev,
                                         tol_real=dup_tol_real,
                                         tol_imag=dup_tol_imag)
            status_str = "SKIP (duplicate)" if already_known else "NEW"

            if k.imag != 0.0:
                Print(" %9f%+9f j    %12g    %s" % (k.real, k.imag, error, status_str))
            else:
                Print(" %12f             %12g    %s" % (k.real, error, status_str))

            if already_known:
                skipped += 1
                continue

            # ── Save eigenvector ─────────────────────────────────────────────
            eigvecfile = os.path.join(results_dir, f'eigf_{file_idx}.pval')
            scatter, eigenvec = PETSc.Scatter.toZero(xr)
            scatter.scatter(xr, eigenvec, False, PETSc.Scatter.Mode.FORWARD)
            if rank == 0:
                mode2pval(eigvecfile, eigenvec, nvars, n, neq, beta, dreduced, rgid)
                if beta != 0:
                    mode2pval3D(eigvecfile, eigenvec, nvars, n, neq, beta, 21, dreduced, rgid)

            new_eigs.append(k)
            file_idx += 1

        Print("")

    # ── Append new eigenvalues to file ───────────────────────────────────────
    if rank == 0:
        n_new = len(new_eigs)
        Print(f' Summary: {nconv} converged  |  {skipped} duplicates skipped  |  {n_new} new')

        if n_new > 0:
            # Index offset: continue from where previous run left off
            write_offset = n_prev
            print(f' Appending {n_new} new eigenvalue(s) to {eigv_file}')
            with open(eigv_file, 'a') as w:
                for j, eig in enumerate(new_eigs):
                    w.write('{0:2d}   {1:12.8f}   {2:12.8f}\n'.format(
                        write_offset + j, eig.real, eig.imag))
            print(' DONE')
        else:
            print(' No new eigenvalues to save.')
        print('')


if __name__ == "__main__":
    run_slices()
