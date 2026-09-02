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
from input_output import openjacobian, read_coordinates, open_sod2d_jacobian, read_sod2d_coordinates, dump_sod2d_coordinates

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
            sensitivity     : compute structural sensitivity?
                              requires adjoint=True (default: False)

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
    p['sod2d_dumpfile'] = cfg.get('io', 'sod2d_dumpfile', fallback='samg.matrix.coo').strip()
    p['simulator']   = cfg.get('io', 'simulator', fallback='tau').strip()

    # [physics]
    p['mach']    = cfg.getfloat('physics', 'mach')
    p['beta']    = cfg.getfloat('physics', 'beta',    fallback=0.0)
    p['rlength'] = cfg.getfloat('physics', 'rlength', fallback=1.0)

    # [solver]
    p['nev']     = cfg.getint('solver', 'nev')
    p['ncv']     = cfg.getint('solver', 'ncv', fallback=0)  # 0 = auto (nev*3+1)
    if cfg.has_option('solver', 'shift_real'):
        sr         = cfg.getfloat('solver', 'shift_real')
        si         = cfg.getfloat('solver', 'shift_imag', fallback=0.0)
        p['shift'] = complex(sr, si)
    elif cfg.has_option('solver', 'shift'):
        p['shift'] = complex(cfg.get('solver', 'shift').strip())
    else:
        raise ValueError('El .ini debe definir shift_real/shift_imag o shift en [solver]')
    p['tol']         = cfg.getfloat  ('solver', 'tol',         fallback=1e-8)
    p['max_it']      = cfg.getint    ('solver', 'max_it',      fallback=15000)
    p['adjoint']     = cfg.getboolean('solver', 'adjoint',     fallback=False)
    p['gen']         = cfg.getboolean('solver', 'gen',         fallback=False)
    p['sensitivity'] = cfg.getboolean('solver', 'sensitivity', fallback=False)

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


def next_eigvec_index(results_dir, prefix='eigf_'):
    """Return the next available index for eigf_N.pval (or other prefix) files."""
    max_idx = -1
    if os.path.isdir(results_dir):
        for fname in os.listdir(results_dir):
            if fname.startswith(prefix) and fname.endswith('.pval'):
                try:
                    idx = int(fname[len(prefix):-len('.pval')])
                    max_idx = max(max_idx, idx)
                except ValueError:
                    pass
    return max_idx + 1


# ─────────────────────────────────────────────────────────────────────────────
# Eigenproblem solver (direct or adjoint)
# ─────────────────────────────────────────────────────────────────────────────

def solve_eigenproblem(A, B, nev, ncv, the_shift, tol, max_it, gen,
                       two_sided, rank):
    """
    Set up and solve the eigenvalue problem A x = λ x using SLEPc/MUMPS
    with shift-invert spectral transformation.

    When two_sided=True, SLEPc also computes the left eigenvectors (adjoint
    modes) in the same solve via E.setTwoSided(True).  The left eigenvector
    y_i satisfies  y_i^H A = λ_i y_i^H, i.e. A^H y_i = conj(λ_i) y_i,
    which is the adjoint problem — retrieved afterwards with
    E.getLeftEigenvector(i, yr, yi).  No matrix transpose is needed.

    Parameters
    ----------
    A          : PETSc.Mat — Jacobian (not modified)
    B          : PETSc.Mat — mass matrix (used only when gen=True)
    nev        : int       — number of eigenvalues requested
    ncv        : int       — number of Krylov vectors (must be > nev)
    the_shift  : complex
    tol, max_it: float, int
    gen        : bool      — generalised EVP?
    two_sided  : bool      — also compute left eigenvectors (adjoint modes)?
    rank       : int       — MPI rank

    Returns
    -------
    E : SLEPc.EPS — solved EPS object (caller must destroy)
    """
    Print = PETSc.Sys.Print

    mpd = ncv - 1

    E = SLEPc.EPS().create()

    if gen:
        Print('  Generalised EGVP  Ax = s M x')
        E.setOperators(A, B)
    else:
        Print('  Standard EGVP  Ax = s x')
        E.setOperators(A)
        E.setProblemType(SLEPc.EPS.ProblemType.NHEP)

    # When two_sided=True SLEPc computes left eigenvectors (adjoint modes)
    # in the same factorisation — no extra solve or matrix transpose needed.
    if two_sided:
        E.setTwoSided(True)

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
        print('')

    E.solve()
    return E


# ─────────────────────────────────────────────────────────────────────────────
# Structural sensitivity
# ─────────────────────────────────────────────────────────────────────────────

def compute_structural_sensitivity(dir_vecs, adj_vecs, B,
                                   nvars, n, neq, beta, dreduced, rgid,
                                   output_path, rank):
    """
    Compute and save the structural sensitivity for each mode pair.

    Definition (Giannetti & Luchini 2007, Hill 1992):

        S_i(x) = || q_adj_i(x) || * || q_dir_i(x) ||
                 ─────────────────────────────────────
                   | <q_adj_i, B q_dir_i> |

    where  q_dir_i  is the i-th right eigenvector (direct mode)
    and    q_adj_i  is the i-th left eigenvector  (adjoint mode),
    both gathered to rank 0 as sequential arrays.

    The denominator normalises by the biorthogonality inner product so that
    the sensitivity amplitude is independent of the arbitrary scalings of the
    two eigenvectors.

    Parameters
    ----------
    dir_vecs  : list of PETSc.Vec  — direct modes in full space (size n)
    adj_vecs  : list of PETSc.Vec  — adjoint modes in full space (size n)
    B         : PETSc.Mat          — mass matrix
    output_path : str
    rank      : int
    """
    Print = PETSc.Sys.Print
    Print('')
    Print(' ── Computing structural sensitivity ──')

    nmodes = min(len(dir_vecs), len(adj_vecs))

    for i in range(nmodes):
        q_dir = dir_vecs[i]
        q_adj = adj_vecs[i]

        # B * q_dir
        Bq_dir, _ = B.getVecs()
        B.mult(q_dir, Bq_dir)

        # Gather to rank 0
        sc_d, dir_seq  = PETSc.Scatter.toZero(q_dir)
        sc_a, adj_seq  = PETSc.Scatter.toZero(q_adj)
        sc_b, Bqd_seq  = PETSc.Scatter.toZero(Bq_dir)

        sc_d.scatter(q_dir,  dir_seq,  False, PETSc.Scatter.Mode.FORWARD)
        sc_a.scatter(q_adj,  adj_seq,  False, PETSc.Scatter.Mode.FORWARD)
        sc_b.scatter(Bq_dir, Bqd_seq,  False, PETSc.Scatter.Mode.FORWARD)

        if rank == 0:
            dir_arr = dir_seq.getArray().copy()   # complex128, size n
            adj_arr = adj_seq.getArray().copy()
            Bqd_arr = Bqd_seq.getArray().copy()

            # Biorthogonality inner product: <q_adj | B | q_dir>
            inner_prod = np.dot(np.conj(adj_arr), Bqd_arr)
            norm_ip    = abs(inner_prod)

            if norm_ip < 1.0e-30:
                Print(' WARNING: mode {0} — adjoint/direct inner product near'
                      ' zero; sensitivity not normalised'.format(i))
                norm_ip = 1.0

            # Pointwise structural sensitivity
            sensitivity = np.abs(adj_arr) * np.abs(dir_arr) / norm_ip

            # Pack as complex array for mode2pval (imaginary part = 0)
            sens_c = sensitivity.astype(np.complex128)

            # Write to a temporary sequential PETSc Vec for mode2pval
            sens_pvec = PETSc.Vec().createSeq(len(sens_c))
            sens_pvec.setArray(sens_c)

            outfile = os.path.join(output_path,
                                   'sensitivity_{0}.pval'.format(i))
            mode2pval(outfile, sens_pvec, nvars, n, neq, beta, dreduced, rgid)
            if beta != 0:
                mode2pval3D(outfile, sens_pvec, nvars, n, neq, beta, 21,
                            dreduced, rgid)

            Print(' Mode {0:3d}  |<q+,Bq>| = {1:.4e}   -> {2}'.format(
                  i, norm_ip, outfile))
            sens_pvec.destroy()

        Bq_dir.destroy()

    Print(' ── Structural sensitivity done ──')
    Print('')


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
    simulator    = params['simulator']
    mach         = params['mach']
    beta         = params['beta']
    rlength      = params['rlength']
    nev          = params['nev']
    _ncv_param   = params['ncv']
    ncv          = _ncv_param if _ncv_param > 0 else nev * 3 + 1  # auto if 0
    the_shift    = params['shift']
    tol          = params['tol']
    max_it       = params['max_it']
    adjoint      = params['adjoint']
    gen          = params['gen']
    sensitivity  = params['sensitivity']
    dreduced     = params['dreduced']
    xmin         = params['xmin'];  xmax = params['xmax']
    zmin         = params['zmin'];  zmax = params['zmax']
    dup_tol_real = params['dup_tol_real']
    dup_tol_imag = params['dup_tol_imag']

    is_simulator_sod2d = (simulator.replace(" ","") == 'sod2d')

    # Sensitivity requires both direct and adjoint modes
    if sensitivity and not adjoint:
        if rank == 0:
            print(' WARNING: sensitivity=True requires adjoint=True.'
                  ' Enabling adjoint automatically.')
        adjoint = True

    jacfile = os.path.join(input_path, params['jac_file'])
    volfile = os.path.join(input_path, params['vol_file'])
    coofile = os.path.join(input_path, params['coord_file'])

    sod2d_dumpfile = os.path.join(output_path, params['sod2d_dumpfile'])


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
    Print(' ncv         : {0}'.format(ncv))
    Print(' Adjoint     : {0}'.format(adjoint))
    Print(' Sensitivity : {0}'.format(sensitivity))
    Print(' Generalised : {0}'.format(gen))
    Print(' Dom. reduc. : {0}'.format(dreduced))
    Print(' MPI ranks   : {0}'.format(nproc))
    Print(' OMP threads : {0}'.format(os.environ.get('OMP_NUM_THREADS', '1')))
    Print(' ========================================')
    Print('')

    # ── Checkpoint files ─────────────────────────────────────────────────────
    eigv_dir_file = os.path.join(output_path, 'eigv_DIR.dat')
    eigv_adj_file = os.path.join(output_path, 'eigv_ADJ.dat')

    if rank == 0:
        eigs_dir_prev = load_previous_eigenvalues(eigv_dir_file)
        eigs_adj_prev = load_previous_eigenvalues(eigv_adj_file)
        n_dir_prev    = len(eigs_dir_prev)
        n_adj_prev    = len(eigs_adj_prev)
        Print(' Found {0} previous direct eigenvalue(s)'.format(n_dir_prev))
        Print(' Found {0} previous adjoint eigenvalue(s)'.format(n_adj_prev))
    else:
        eigs_dir_prev = eigs_adj_prev = None
        n_dir_prev    = n_adj_prev    = None

    eigs_dir_prev = comm.bcast(eigs_dir_prev, root=0)
    eigs_adj_prev = comm.bcast(eigs_adj_prev, root=0)
    n_dir_prev    = comm.bcast(n_dir_prev,    root=0)
    n_adj_prev    = comm.bcast(n_adj_prev,    root=0)

    dir_file_start = next_eigvec_index(output_path, prefix='eigf_')
    adj_file_start = next_eigvec_index(output_path, prefix='eiga_')

    # ── Read Jacobian ────────────────────────────────────────────────────────
    t0 = time.time()
    Print(' Reading Jacobian')
    if rank == 0:
        if(is_simulator_sod2d):
            amatrix, neq = open_sod2d_jacobian(jacfile)
        else:
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

    Print(' Matrix main dimension = {0}'.format(nvars))
    Print(' Number of equations   = {0}'.format(neq))
    Print('')
    _t(comm, rank, 'Jacobian read', t0)

    gridpoints = int(nvars / neq)

    # ── Mass matrix ──────────────────────────────────────────────────────────
    t0 = time.time()
    Print(' Reading mass matrix and generating M')
    Print('')
    if rank == 0:
        if(is_simulator_sod2d):
            vols_buf = np.repeat(1,nvars).astype(np.float64)
        else:
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

    # TODO had a 0.01 multiplying identity mat here - find out why
    bmatrix = identity(nvars, dtype='c16', format='csr')
    if(not is_simulator_sod2d):
        bmatrix.data[:] = np.repeat(vols_buf, neq).astype(np.complex128)
    _t(comm, rank, 'Mass matrix build', t0)

    # ── Domain reduction — rank 0 only, then broadcast ──────────────────────
    t0 = time.time()
    if dreduced:
        Print(' Applying domain reduction')
        Print(' XMIN/XMAX = {0}/{1}'.format(xmin, xmax))
        Print(' ZMIN/ZMAX = {0}/{1}'.format(zmin, zmax))

        if rank == 0:
            if(is_simulator_sod2d):
                coord = read_sod2d_coordinates(coofile, rlength, beta)
            else:
                coord = read_coordinates(coofile, rlength, beta)
            dr = domain_reduction(zmin, zmax, xmin, xmax)
            dr.create_Pmatrix(coord)

            nnz_before = amatrix.nnz
            amatrix = dr.reduce_matrix(amatrix)
            bmatrix = dr.reduce_matrix(bmatrix)
            n_red   = amatrix.shape[0]
            nnz_red = amatrix.nnz

            localid = np.arange(0, gridpoints, 1, dtype='i4')
            localid = np.repeat(localid, neq)
            rgid    = dr.reduce_vector(localid)[0::neq].astype(int)

            Print(' Previous NNZ = {0}'.format(nnz_before))
            Print(' New NNZ      = {0}'.format(nnz_red))
            Print(' New leading dimension of A = {0}'.format(n_red))
            Print('')

            # Pack reduced amatrix for broadcast
            a_indptr  = amatrix.indptr.astype(np.int32)
            a_indices = amatrix.indices.astype(np.int32)
            a_data    = amatrix.data.astype(np.complex128)
            b_data    = bmatrix.data.astype(np.complex128)  # diagonal only
            meta_dr   = np.array([n_red, amatrix.nnz, len(rgid)], dtype=np.int64)
        else:
            meta_dr   = np.empty(3, dtype=np.int64)
            a_indptr  = None
            a_indices = None
            a_data    = None
            b_data    = None
            rgid      = None

        # Broadcast metadata
        comm.Bcast(meta_dr, root=0)
        n_red, nnz_red, ngrid_red = int(meta_dr[0]), int(meta_dr[1]), int(meta_dr[2])

        # Broadcast sparse arrays
        if rank != 0:
            a_indptr  = np.empty(n_red + 1,  dtype=np.int32)
            a_indices = np.empty(nnz_red,     dtype=np.int32)
            a_data    = np.empty(nnz_red,     dtype=np.complex128)
            b_data    = np.empty(n_red,       dtype=np.complex128)
            rgid      = np.empty(ngrid_red,   dtype=np.int64)

        comm.Bcast(a_indptr,  root=0)
        comm.Bcast(a_indices, root=0)
        comm.Bcast(a_data,    root=0)
        comm.Bcast(b_data,    root=0)
        comm.Bcast(rgid,      root=0)

        # Reconstruct scipy matrices on all ranks
        amatrix = csr_matrix((a_data, a_indices, a_indptr), shape=(n_red, n_red))
        bmatrix = identity(n_red, dtype='c16', format='csr')
        bmatrix.data[:] = b_data

        n    = n_red
        rgid = rgid.astype(int)
        # ────── Dump coords to .coo if using SOD2D ───────────────────────────────
        if(is_simulator_sod2d and rank==0):
            coord = read_sod2d_coordinates(coofile, rlength, beta)
            dump_sod2d_coordinates(sod2d_dumpfile, coord, 5, dr.kept_idx)
    else:
        rgid = None
        n    = nvars
        if(is_simulator_sod2d and rank==0):
            coord = read_sod2d_coordinates(coofile, rlength, beta)
            dump_sod2d_coordinates(sod2d_dumpfile, coord, 5)
    _t(comm, rank, 'Domain reduction', t0)



    # ── Assemble PETSc matrices ──────────────────────────────────────────────
    t0 = time.time()
    A = PETSc.Mat()
    A.create(PETSc.COMM_WORLD)
    A.setSizes([n, n])
    A.setFromOptions()
    A.setUp()

    Print(' Assembling PETSc matrix A...')
    rstart, rend  = A.getOwnershipRange()
    indptr_local  = amatrix.indptr[rstart:rend + 1].copy()
    indices_local = amatrix.indices[indptr_local[0]:indptr_local[-1]].copy()
    values_local  = amatrix.data   [indptr_local[0]:indptr_local[-1]].copy()
    indptr_local  = (indptr_local - indptr_local[0]).astype(PETSc.IntType)
    indices_local = indices_local.astype(PETSc.IntType)
    values_local  = values_local.astype(PETSc.ScalarType)
    A.setValuesCSR(indptr_local, indices_local, values_local)
    A.assemble()
    _t(comm, rank, 'PETSc matrix A assembly', t0)

    t0 = time.time()
    B = PETSc.Mat()
    B.create(PETSc.COMM_WORLD)
    B.setSizes([n, n])
    B.setFromOptions()
    B.setUp()
    rstart, rend = B.getOwnershipRange()
    diag_vals    = bmatrix.data[rstart:rend].astype(PETSc.ScalarType)
    diag_vec     = PETSc.Vec().createWithArray(diag_vals, comm=PETSc.COMM_WORLD)
    B.setDiagonal(diag_vec)
    B.assemble()
    _t(comm, rank, 'PETSc matrix B assembly', t0)

    # ─────────────────────────────────────────────────────────────────────────
    # DIRECT PROBLEM   A x = λ x
    # ─────────────────────────────────────────────────────────────────────────
    Print('\n################################')
    if adjoint:
        Print('    SOLVING DIRECT + ADJOINT PROBLEM')
    else:
        Print('    SOLVING DIRECT PROBLEM')
    Print('################################\n')

    t0 = time.time()
    E_dir = solve_eigenproblem(A, B, nev, ncv, the_shift, tol, max_it,
                               gen, two_sided=adjoint, rank=rank)
    _t(comm, rank, 'EPS solve', t0)

    its_dir   = E_dir.getIterationNumber()
    nconv_dir = E_dir.getConverged()

    if rank == 0:
        print('')
        print(' Iterations (EPS)        : ', its_dir)
        print(' Converged               : ', nconv_dir)

    xr, _ = A.getVecs()
    xi, _ = A.getVecs()

    new_dir_eigs  = []
    dir_vecs_kept = []   # PETSc Vecs for sensitivity (only new modes)
    skipped_dir   = 0
    dir_file_idx  = dir_file_start

    t0 = time.time()
    if nconv_dir > 0:
        Print("")
        Print(" DIRECT MODES")
        Print("           k           ||Ax-kx||/||kx||   status")
        Print("----------------------------------------------------")

        for i in range(nconv_dir):
            k     = E_dir.getEigenpair(i, xr, xi)
            error = E_dir.computeError(i)

            already_known = is_duplicate(k, eigs_dir_prev,
                                         tol_real=dup_tol_real,
                                         tol_imag=dup_tol_imag)
            status_str = "SKIP (duplicate)" if already_known else "NEW"

            if k.imag != 0.0:
                Print(" %9f%+9f j    %12g    %s" % (k.real, k.imag, error, status_str))
            else:
                Print(" %12f             %12g    %s" % (k.real, error, status_str))

            if already_known:
                skipped_dir += 1
                continue

            # Save eigenvector file
            eigvecfile = os.path.join(output_path,
                                      'eigf_{0}.pval'.format(dir_file_idx))
            scatter, eigenvec = PETSc.Scatter.toZero(xr)
            scatter.scatter(xr, eigenvec, False, PETSc.Scatter.Mode.FORWARD)
            if rank == 0:
                mode2pval(eigvecfile, eigenvec, nvars, n, neq, beta, dreduced, rgid, is_simulator_sod2d)
                if beta != 0:
                    mode2pval3D(eigvecfile, eigenvec, nvars, n, neq, beta, 21,
                                dreduced, rgid)

            # Keep a copy in memory for sensitivity computation
            if sensitivity:
                mode_copy, _ = A.getVecs()
                xr.copy(mode_copy)
                dir_vecs_kept.append(mode_copy)

            new_dir_eigs.append(k)
            dir_file_idx += 1

        Print("")
    _t(comm, rank, 'Direct eigenvector save', t0)

    # Append direct eigenvalues to checkpoint file
    if rank == 0:
        n_new_dir = len(new_dir_eigs)
        Print(' Direct summary: {0} converged | {1} skipped | {2} new'.format(
              nconv_dir, skipped_dir, n_new_dir))
        if n_new_dir > 0:
            print(' Appending direct eigenvalues -> ', eigv_dir_file)
            with open(eigv_dir_file, 'a') as w:
                for j, eig in enumerate(new_dir_eigs):
                    w.write('{0:2d}   {1:12.8f}   {2:12.8f}\n'.format(
                        n_dir_prev + j, eig.real, eig.imag))
            print(' DONE')
        print('')

    # ── Adjoint modes are the left eigenvectors from the same solve ──────────
    # (two_sided=adjoint was passed above; left vecs available via
    #  E_dir.getLeftEigenvector when adjoint=True)
    if adjoint:
        nconv_adj = E_dir.getConverged()

        if rank == 0:
            print('')
            print(' Converged (adjoint modes) : ', nconv_adj)

        # Left eigenvectors retrieved from the same EPS object
        yr, _ = A.getVecs()
        yi, _ = A.getVecs()

        new_adj_eigs  = []
        adj_vecs_kept = []   # PETSc Vecs for sensitivity
        skipped_adj   = 0
        adj_file_idx  = adj_file_start

        t0 = time.time()
        if nconv_adj > 0:
            Print("")
            Print(" ADJOINT MODES  (left eigenvectors: A^H y = conj(lambda) y)")
            Print("           k*          ||A^H y - k* y||   status")
            Print("----------------------------------------------------")

            for i in range(nconv_adj):
                # Right eigenvalue is λ; left eigenvector satisfies A^H y = λ* y
                k = E_dir.getEigenpair(i, xr, xi)        # reuse xr/xi (right, not saved)
                E_dir.getLeftEigenvector(i, yr, yi)       # adjoint mode

                error = E_dir.computeError(i)

                # Eigenvalue for the adjoint mode is conj(k)
                k_adj = k.conjugate()

                already_known = is_duplicate(k_adj, eigs_adj_prev,
                                             tol_real=dup_tol_real,
                                             tol_imag=dup_tol_imag)
                status_str = "SKIP (duplicate)" if already_known else "NEW"

                if k_adj.imag != 0.0:
                    Print(" %9f%+9f j    %12g    %s" % (
                          k_adj.real, k_adj.imag, error, status_str))
                else:
                    Print(" %12f             %12g    %s" % (
                          k_adj.real, error, status_str))

                if already_known:
                    skipped_adj += 1
                    continue

                # Save adjoint eigenvector file (prefix eiga_)
                adjvecfile = os.path.join(output_path,
                                          'eiga_{0}.pval'.format(adj_file_idx))
                scatter_a, adjvec = PETSc.Scatter.toZero(yr)
                scatter_a.scatter(yr, adjvec, False, PETSc.Scatter.Mode.FORWARD)
                if rank == 0:
                    mode2pval(adjvecfile, adjvec, nvars, n, neq, beta, dreduced, rgid)
                    if beta != 0:
                        mode2pval3D(adjvecfile, adjvec, nvars, n, neq, beta, 21,
                                    dreduced, rgid)

                # Keep a copy for sensitivity
                if sensitivity:
                    adj_copy, _ = A.getVecs()
                    yr.copy(adj_copy)
                    adj_vecs_kept.append(adj_copy)

                new_adj_eigs.append(k_adj)
                adj_file_idx += 1

            Print("")
        _t(comm, rank, 'Adjoint modes save', t0)


        yr.destroy(); yi.destroy()

        # Append adjoint eigenvalues
        if rank == 0:
            n_new_adj = len(new_adj_eigs)
            Print(' Adjoint summary: {0} converged | {1} skipped | {2} new'.format(
                  nconv_adj, skipped_adj, n_new_adj))
            if n_new_adj > 0:
                print(' Appending adjoint eigenvalues -> ', eigv_adj_file)
                with open(eigv_adj_file, 'a') as w:
                    for j, eig in enumerate(new_adj_eigs):
                        w.write('{0:2d}   {1:12.8f}   {2:12.8f}\n'.format(
                            n_adj_prev + j, eig.real, eig.imag))
                print(' DONE')
            print('')

        # ── Structural sensitivity ────────────────────────────────────────────
        if sensitivity and len(dir_vecs_kept) > 0 and len(adj_vecs_kept) > 0:
            t0 = time.time()
            compute_structural_sensitivity(
                dir_vecs_kept, adj_vecs_kept, B,
                nvars, n, neq, beta, dreduced, rgid,
                output_path, rank)
            _t(comm, rank, 'Structural sensitivity', t0)
        elif sensitivity:
            Print(' WARNING: no mode pairs available for sensitivity computation.')

        # Clean up stored mode copies
        for v in dir_vecs_kept + adj_vecs_kept:
            v.destroy()

    E_dir.destroy()
    xr.destroy(); xi.destroy()

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