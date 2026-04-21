#! /usr/bin/env python
#
# Usage: python EIGENSOLVER.py eigensolver.ini
#        mpirun -n 4 python EIGENSOLVER.py eigensolver.ini
#
import numpy as np
import sys, os
import configparser
from netCDF4 import Dataset
from scipy.sparse import csr_matrix, linalg as sla
from scipy.sparse import identity

from jac_red import domain_reduction
from save2pval import mode2pval, mode2pval3D
from input_output import openjacobian, read_coordinates

import petsc4py
import slepc4py
slepc4py.init(sys.argv)
from petsc4py import PETSc
from slepc4py import SLEPc
from mpi4py import MPI


# ─────────────────────────────────────────────────────────────────────────────
# Control file reader
# ─────────────────────────────────────────────────────────────────────────────

def read_control_file(filepath):
    """
    Read parameters from a .ini control file.

    Expected sections and keys:

        [io]
            input_path      : directory containing Jacobian / volume / coord files
            output_path     : directory for eigenvalue and eigenvector output
            jac_file        : Jacobian filename   (default: samg.matrix.amg.pval)
            vol_file        : volumes filename    (default: samg.matrix.vol)
            coord_file      : coordinates filename (default: samg.matrix.coo)

        [physics]
            mach            : Mach number
            beta            : spanwise wavenumber (0 for 2D)
            rlength         : reference length (default: 1.0)

        [solver]
            nev             : number of eigenvalues requested
            shift_real      : real part of the spectral shift
            shift_imag      : imaginary part of the spectral shift
            tol             : EPS convergence tolerance  (default: 1e-8)
            max_it          : EPS maximum iterations     (default: 15000)
            adjoint         : solve adjoint problem?     (default: False)
            gen             : generalised EVP (Ax=sMx)?  (default: False)

        [domain_reduction]
            enabled         : apply domain reduction?    (default: False)
            xmin            : x lower bound
            xmax            : x upper bound
            zmin            : z lower bound
            zmax            : z upper bound

        [checkpoint]
            dup_tol_real    : duplicate tolerance, real part  (default: 1e-5)
            dup_tol_imag    : duplicate tolerance, imag part  (default: 1e-5)

    Returns
    -------
    dict with all parameters.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError('Control file not found: {0}'.format(filepath))

    cfg = configparser.ConfigParser()
    cfg.read(filepath)

    p = {}

    # [io]
    p['input_path']  = cfg.get('io', 'input_path').strip()
    p['output_path'] = cfg.get('io', 'output_path').strip()
    p['jac_file']    = cfg.get('io', 'jac_file',   fallback='samg.matrix.amg.pval').strip()
    p['vol_file']    = cfg.get('io', 'vol_file',   fallback='samg.matrix.vol').strip()
    p['coord_file']  = cfg.get('io', 'coord_file', fallback='samg.matrix.coo').strip()

    # [physics]
    p['mach']    = cfg.getfloat('physics', 'mach')
    p['beta']    = cfg.getfloat('physics', 'beta',    fallback=0.0)
    p['rlength'] = cfg.getfloat('physics', 'rlength', fallback=1.0)

    # [solver]
    p['nev']     = cfg.getint  ('solver', 'nev')
    sr           = cfg.getfloat('solver', 'shift_real')
    si           = cfg.getfloat('solver', 'shift_imag')
    p['shift']   = complex(sr, si)
    p['tol']     = cfg.getfloat('solver', 'tol',     fallback=1e-8)
    p['max_it']  = cfg.getint  ('solver', 'max_it',  fallback=15000)
    p['adjoint'] = cfg.getboolean('solver', 'adjoint', fallback=False)
    p['gen']     = cfg.getboolean('solver', 'gen',     fallback=False)

    # [domain_reduction]
    p['dreduced'] = cfg.getboolean('domain_reduction', 'enabled', fallback=False)
    p['xmin']     = cfg.getfloat  ('domain_reduction', 'xmin',    fallback=0.0)
    p['xmax']     = cfg.getfloat  ('domain_reduction', 'xmax',    fallback=1.0)
    p['zmin']     = cfg.getfloat  ('domain_reduction', 'zmin',    fallback=-1.0)
    p['zmax']     = cfg.getfloat  ('domain_reduction', 'zmax',    fallback=1.0)

    # [checkpoint]
    p['dup_tol_real'] = cfg.getfloat('checkpoint', 'dup_tol_real', fallback=1e-5)
    p['dup_tol_imag'] = cfg.getfloat('checkpoint', 'dup_tol_imag', fallback=1e-5)

    return p


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_previous_eigenvalues(eigv_file):
    """Read eigenvalues already stored in eigv_file.
    Returns a list of complex numbers and the count of entries found."""
    eigs_prev = []
    if not os.path.isfile(eigv_file):
        return eigs_prev
    with open(eigv_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            re = float(parts[1])
            im = float(parts[2])
            eigs_prev.append(complex(re, im))
    return eigs_prev


def is_duplicate(new_eig, existing_eigs, tol_real=1e-5, tol_imag=1e-5):
    """Return True if new_eig already exists in existing_eigs within tolerance."""
    for e in existing_eigs:
        if abs(new_eig.real - e.real) < tol_real and \
           abs(new_eig.imag - e.imag) < tol_imag:
            return True
    return False


def next_eigvec_index(results_dir):
    """Return the next available index for eigf_N.pval files."""
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

def run_slices(params):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    Print = PETSc.Sys.Print

    # ── Unpack parameters ────────────────────────────────────────────────────
    input_path   = params['input_path']
    output_path  = params['output_path']
    mach         = params['mach']
    beta         = params['beta']
    rlength      = params['rlength']
    nev          = params['nev']
    the_shift    = params['shift']
    tol          = params['tol']
    max_it       = params['max_it']
    adjoint      = params['adjoint']
    gen          = params['gen']
    dreduced     = params['dreduced']
    xmin         = params['xmin'];  xmax = params['xmax']
    zmin         = params['zmin'];  zmax = params['zmax']
    dup_tol_real = params['dup_tol_real']
    dup_tol_imag = params['dup_tol_imag']

    jacfile = os.path.join(input_path, params['jac_file'])
    volfile = os.path.join(input_path, params['vol_file'])
    coofile = os.path.join(input_path, params['coord_file'])

    fac = 1. / (mach * np.sqrt(1.4))

    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    # ── Print run summary ────────────────────────────────────────────────────
    Print('')
    Print(' ========================================')
    Print('  EIGENSOLVER — FLORES')
    Print(' ========================================')
    Print(' Input path  : {0}'.format(input_path))
    Print(' Output path : {0}'.format(output_path))
    Print(' Jacobian    : {0}'.format(jacfile))
    Print(' Mach        : {0}'.format(mach))
    Print(' beta        : {0}'.format(beta))
    Print(' Shift       : {0}'.format(the_shift))
    Print(' nev         : {0}'.format(nev))
    Print(' Adjoint     : {0}'.format(adjoint))
    Print(' Generalised : {0}'.format(gen))
    Print(' Dom. reduc. : {0}'.format(dreduced))
    Print(' ========================================')
    Print('')

    # ── Determine eigenvalue output file ────────────────────────────────────
    eigv_filename = 'eigv_ADJ.dat' if adjoint else 'eigv_DIR.dat'
    eigv_file     = os.path.join(output_path, eigv_filename)

    # ── Load checkpoint (rank 0, then broadcast) ─────────────────────────────
    if rank == 0:
        eigs_prev = load_previous_eigenvalues(eigv_file)
        n_prev    = len(eigs_prev)
        Print(' Found {0} previously computed eigenvalue(s) in {1}'.format(
              n_prev, eigv_file))
    else:
        eigs_prev = None
        n_prev    = None

    eigs_prev = comm.bcast(eigs_prev, root=0)
    n_prev    = comm.bcast(n_prev,    root=0)
    file_start_idx = next_eigvec_index(output_path)

    # ── Read Jacobian ────────────────────────────────────────────────────────
    Print(' Reading Jacobian')
    amatrix, neq = openjacobian(jacfile)
    amatrix.data *= fac
    nvars = amatrix.shape[0]
    Print(' Matrix main dimension = {0}'.format(nvars))
    Print(' Number of equations   = {0}'.format(neq))
    Print('')

    gridpoints = int(nvars / neq)

    # ── Mass matrix ──────────────────────────────────────────────────────────
    Print(' Reading mass matrix and generating M')
    Print('')
    one = 1. + 0j
    with open(volfile, 'r') as f:
        vols = [float(line) for line in f.readlines()]
    bmatrix = identity(nvars, dtype='c16', format='csr')
    bmatrix.data[:] = [vols[i // neq] * one for i in range(nvars)]

    # ── Domain reduction ─────────────────────────────────────────────────────
    if dreduced:
        Print(' Applying domain reduction')
        Print(' XMIN/XMAX = {0}/{1}'.format(xmin, xmax))
        Print(' ZMIN/ZMAX = {0}/{1}'.format(zmin, zmax))
        coord = read_coordinates(coofile, rlength, beta)
        dr = domain_reduction(zmin, zmax, xmin, xmax)
        dr.create_Pmatrix(coord)
        Print(' Previous NNZ = {0}'.format(amatrix.nnz))
        amatrix = dr.reduce_matrix(amatrix)
        Print(' New NNZ      = {0}'.format(amatrix.nnz))
        bmatrix = dr.reduce_matrix(bmatrix)
        n = amatrix.shape[0]
        Print(' New leading dimension of A = {0}'.format(n))
        localid = np.arange(0, gridpoints, 1, dtype='i4')
        localid = np.repeat(localid, neq)
        rgid = dr.reduce_vector(localid)[0::neq].astype(int)
        Print('')
    else:
        rgid = None
        n    = nvars

    # ── Assemble PETSc matrices ──────────────────────────────────────────────
    A = PETSc.Mat()
    A.create(PETSc.COMM_WORLD)
    A.setSizes([n, n])
    A.setUp()
    A.setType('seqaij')
    Print(' Assembling PETSc matrix A...')
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

    E.setTolerances(tol=tol, max_it=max_it)
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
    its             = E.getIterationNumber()
    tol_out, max_it_out = E.getTolerances()
    ksp_its         = K.getIterationNumber()
    nconv           = E.getConverged()

    if rank == 0:
        print('')
        print(' Iterations (EPS)  : ', its)
        print(' Tolerance / max_it: ', tol_out, ' / ', max_it_out)
        print(' Iterations (KSP)  : ', ksp_its)
        print(' Requested         : ', nev)
        print(' Converged         : ', nconv)

    xr, xrl = A.getVecs()
    xi, xil = A.getVecs()

    new_eigs   = []
    skipped    = 0
    file_idx   = file_start_idx

    if nconv > 0:
        Print("")
        Print("           k           ||Ax-kx||/||kx||   status")
        Print("----------------------------------------------------")

        for i in range(nconv):
            k     = E.getEigenpair(i, xr, xi)
            error = E.computeError(i)

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

            # Save eigenvector
            eigvecfile = os.path.join(output_path, 'eigf_{0}.pval'.format(file_idx))
            scatter, eigenvec = PETSc.Scatter.toZero(xr)
            scatter.scatter(xr, eigenvec, False, PETSc.Scatter.Mode.FORWARD)
            if rank == 0:
                mode2pval(eigvecfile, eigenvec, nvars, n, neq, beta, dreduced, rgid)
                if beta != 0:
                    mode2pval3D(eigvecfile, eigenvec, nvars, n, neq, beta, 21, dreduced, rgid)

            new_eigs.append(k)
            file_idx += 1

        Print("")

    # ── Append new eigenvalues to checkpoint file ────────────────────────────
    if rank == 0:
        n_new = len(new_eigs)
        Print(' Summary: {0} converged  |  {1} duplicates skipped  |  {2} new'.format(
              nconv, skipped, n_new))

        if n_new > 0:
            print(' Appending {0} new eigenvalue(s) to {1}'.format(n_new, eigv_file))
            with open(eigv_file, 'a') as w:
                for j, eig in enumerate(new_eigs):
                    w.write('{0:2d}   {1:12.8f}   {2:12.8f}\n'.format(
                        n_prev + j, eig.real, eig.imag))
            print(' DONE')
        else:
            print(' No new eigenvalues to save.')
        print('')


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print('Usage: python EIGENSOLVER.py <control_file.ini>')
        print('       mpirun -n 4 python EIGENSOLVER.py <control_file.ini>')
        sys.exit(1)

    ctrl_file = sys.argv[1]
    params    = read_control_file(ctrl_file)
    run_slices(params)