#! /usr/bin/env python
#
# Usage: python eig_simple.py eigensolver.ini
#        mpirun -n 4 python eig_simple.py eigensolver.ini
#
import numpy as np
import sys, os
import time
import configparser
from scipy.sparse import csr_matrix, linalg as sla, identity

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
# Timing helper
# ─────────────────────────────────────────────────────────────────────────────

def _t(comm, rank, label, t0):
    """Print elapsed time since t0 from rank 0, after a barrier sync."""
    comm.Barrier()
    if rank == 0:
        PETSc.Sys.Print(' [TIMING] {0:<40s} {1:8.2f} s'.format(
            label, time.time() - t0))


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
    # Acepta shift_real/shift_imag separados O shift como complejo directo
    # Ejemplos: shift_real = -0.25 / shift_imag = 2.0
    #           shift = -0.25+2.0j
    if cfg.has_option('solver', 'shift_real'):
        sr         = cfg.getfloat('solver', 'shift_real')
        si         = cfg.getfloat('solver', 'shift_imag', fallback=0.0)
        p['shift'] = complex(sr, si)
    elif cfg.has_option('solver', 'shift'):
        p['shift'] = complex(cfg.get('solver', 'shift').strip())
    else:
        raise ValueError('El .ini debe definir shift_real/shift_imag o shift en [solver]')
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
    """Read eigenvalues already stored in eigv_file."""
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
    comm  = MPI.COMM_WORLD
    rank  = comm.Get_rank()
    nproc = comm.Get_size()
    Print = PETSc.Sys.Print

    t_total = time.time()

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
    dup_tol_real    = params['dup_tol_real']
    dup_tol_imag    = params['dup_tol_imag']


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

    Print(' MPI ranks   : {0}'.format(nproc))
    Print(' OMP threads : {0}'.format(os.environ.get('OMP_NUM_THREADS', '1')))
    Print(' ========================================')
    Print('')

    # ── Determine eigenvalue output file ────────────────────────────────────
    eigv_filename = 'eigv_ADJ.dat' if adjoint else 'eigv_DIR.dat'
    eigv_file     = os.path.join(output_path, eigv_filename)

    # ── Load checkpoint ──────────────────────────────────────────────────────
    t0 = time.time()
    if rank == 0:
        eigs_prev = load_previous_eigenvalues(eigv_file)
        n_prev    = len(eigs_prev)
        Print(' Found {0} previously computed eigenvalue(s) in {1}'.format(
              n_prev, eigv_file))
    else:
        eigs_prev = None
        n_prev    = None

    eigs_prev      = comm.bcast(eigs_prev, root=0)
    n_prev         = comm.bcast(n_prev,    root=0)
    file_start_idx = next_eigvec_index(output_path)
    _t(comm, rank, 'Checkpoint load', t0)

    # ── Read Jacobian (rank 0 only, then Bcast with MPI buffer protocol) ─────
    t0 = time.time()
    Print(' Reading Jacobian')
    if rank == 0:
        amatrix, neq = openjacobian(jacfile)
        amatrix.data *= fac
        nvars  = amatrix.shape[0]
        nnz    = amatrix.nnz
        meta   = np.array([neq, nvars, nnz], dtype=np.int64)
    else:
        meta = np.empty(3, dtype=np.int64)

    comm.Bcast(meta, root=0)
    neq, nvars, nnz = int(meta[0]), int(meta[1]), int(meta[2])

    if rank == 0:
        indptr_buf  = amatrix.indptr.astype(np.int32)
        indices_buf = amatrix.indices.astype(np.int32)
        data_buf    = amatrix.data.astype(np.complex128)
    else:
        indptr_buf  = np.empty(nvars + 1, dtype=np.int32)
        indices_buf = np.empty(nnz,       dtype=np.int32)
        data_buf    = np.empty(nnz,       dtype=np.complex128)

    comm.Bcast(indptr_buf,  root=0)
    comm.Bcast(indices_buf, root=0)
    comm.Bcast(data_buf,    root=0)

    amatrix = csr_matrix((data_buf, indices_buf, indptr_buf),
                         shape=(nvars, nvars))

    # Variables locales para ensamblado de PETSc (calculadas post-Bcast)
    # Se actualizan tras domain_reduction si aplica
    _amatrix_indptr_local  = None
    _amatrix_indices_local = None
    _amatrix_data_local    = None

    Print(' Matrix main dimension = {0}'.format(nvars))
    Print(' Number of equations   = {0}'.format(neq))
    Print('')
    _t(comm, rank, 'Jacobian read', t0)

    gridpoints = int(nvars / neq)

    # ── Mass matrix — vectorizado con numpy (sin list comprehension) ─────────
    t0 = time.time()
    Print(' Reading mass matrix and generating M')
    Print('')
    if rank == 0:
        with open(volfile, 'r') as f:
            vols_buf = np.array([float(line) for line in f.readlines()],
                                dtype=np.float64)
        ngp = np.array([len(vols_buf)], dtype=np.int64)
    else:
        ngp = np.empty(1, dtype=np.int64)

    comm.Bcast(ngp, root=0)
    if rank != 0:
        vols_buf = np.empty(int(ngp[0]), dtype=np.float64)
    comm.Bcast(vols_buf, root=0)

    # Vectorizado: np.repeat es O(n) en C, 10-50x mas rapido que list comprehension
    bmatrix = identity(nvars, dtype='c16', format='csr')
    bmatrix.data[:] = np.repeat(vols_buf, neq).astype(np.complex128)
    _t(comm, rank, 'Mass matrix build', t0)

    # ── Domain reduction ─────────────────────────────────────────────────────
    t0 = time.time()
    if dreduced:
        # domain_reduction necesita la matriz completa — ya fue reconstruida
        # en la seccion de Jacobian read cuando dreduced=True
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
    _t(comm, rank, 'Domain reduction', t0)

    # ── Assemble PETSc matrix A (setValuesCSR — bulk insert) ─────────────────
    t0 = time.time()

    # Nota: MUMPS lee OMP_NUM_THREADS del entorno automaticamente.
    # No se pasan opciones mat_mumps_* desde Python para evitar dependencia
    # de SLURM y el problema de "unused options" en PETSc 3.25.
    # Configura el paralelismo via variables de entorno en el job script:
    #   export OMP_NUM_THREADS=N
    #   export OMP_PROC_BIND=close
    #   export OMP_PLACES=cores

    A = PETSc.Mat()
    A.create(PETSc.COMM_WORLD)
    A.setSizes([n, n])
    A.setFromOptions()
    A.setUp()

    Print(' Assembling PETSc matrix A...')
    rstart, rend  = A.getOwnershipRange()
    indptr_local  = amatrix.indptr[rstart:rend + 1].copy()
    indices_local = amatrix.indices[indptr_local[0]:indptr_local[-1]].copy()
    values_local  = amatrix.data[indptr_local[0]:indptr_local[-1]].copy()
    indptr_local  = (indptr_local - indptr_local[0]).astype(PETSc.IntType)
    indices_local = indices_local.astype(PETSc.IntType)
    values_local  = values_local.astype(PETSc.ScalarType)
    A.setValuesCSR(indptr_local, indices_local, values_local)
    A.assemble()
    _t(comm, rank, 'PETSc matrix A assembly', t0)

    # ── Assemble PETSc matrix B — vectorizado con setDiagonal ────────────────
    t0 = time.time()
    B = PETSc.Mat()
    B.create(PETSc.COMM_WORLD)
    B.setSizes([n, n])
    B.setFromOptions()
    B.setUp()

    # setDiagonal es una sola llamada C en lugar de un bucle Python fila a fila
    rstart, rend = B.getOwnershipRange()
    diag_vals    = bmatrix.data[rstart:rend].astype(PETSc.ScalarType)
    diag_vec     = PETSc.Vec().createWithArray(diag_vals, comm=PETSc.COMM_WORLD)
    B.setDiagonal(diag_vec)
    B.assemble()
    _t(comm, rank, 'PETSc matrix B assembly', t0)

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
        print(' OMP threads (OMP_NUM_THREADS)   = ', os.environ.get('OMP_NUM_THREADS', '1'))
        print('')

    # ── EPS solve ────────────────────────────────────────────────────────────
    t0 = time.time()
    E.solve()
    _t(comm, rank, 'EPS solve', t0)

    # ── Post-processing ──────────────────────────────────────────────────────
    its                  = E.getIterationNumber()
    tol_out, max_it_out  = E.getTolerances()
    nconv                = E.getConverged()

    if rank == 0:
        print('')
        print(' Iterations (EPS)  : ', its)
        print(' Tolerance / max_it: ', tol_out, ' / ', max_it_out)
        print(' Requested         : ', nev)
        print(' Converged         : ', nconv)

    xr, xrl = A.getVecs()
    xi, xil = A.getVecs()

    new_eigs = []
    skipped  = 0
    file_idx = file_start_idx

    # ── Save eigenvectors ────────────────────────────────────────────────────
    t0 = time.time()
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

            eigvecfile = os.path.join(output_path, 'eigf_{0}.pval'.format(file_idx))
            scatter, eigenvec = PETSc.Scatter.toZero(xr)
            scatter.scatter(xr, eigenvec, False, PETSc.Scatter.Mode.FORWARD)
            if rank == 0:
                mode2pval(eigvecfile, eigenvec, nvars, n, neq, beta, dreduced, rgid)
                if beta != 0:
                    mode2pval3D(eigvecfile, eigenvec, nvars, n, neq, beta, 21,
                                dreduced, rgid)

            new_eigs.append(k)
            file_idx += 1

        Print("")
    _t(comm, rank, 'Eigenvector save', t0)

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

    # ── Total ────────────────────────────────────────────────────────────────
    _t(comm, rank, 'TOTAL', t_total)


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print('Usage: python eig_simple.py <control_file.ini>')
        print('       mpirun -n 4 python eig_simple.py <control_file.ini>')
        sys.exit(1)

    ctrl_file = sys.argv[1]
    params    = read_control_file(ctrl_file)
    run_slices(params)