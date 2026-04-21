#! /usr/bin/env python
#
# Usage: python RESOLVANT.py resolvent.ini
#        mpirun -n 4 python RESOLVANT.py resolvent.ini
#
import numpy as np
import sys, os
import configparser
from netCDF4 import Dataset
from scipy.sparse import csr_matrix, csc_matrix, linalg as sla
from scipy.sparse import identity
from jac_red import domain_reduction
from save2pval import mode2pval
from input_output import openjacobian, read_coordinates, openqe
import pdb
import petsc4py
import slepc4py
slepc4py.init(sys.argv)
from petsc4py import PETSc
from slepc4py import SLEPc
from mpi4py import MPI

####################################################################
## CONTROL FILE READER
####################################################################

def read_control_file(filepath):
    """
    Read parameters from a .ini control file.

    Expected sections and keys:
        [io]
            input_path          : path to Jacobian and volume files
            output_path         : path for output files

        [physics]
            mach                : Mach number
            beta                : spanwise wavenumber (0 for 2D)
            nslices             : number of slices (only used if beta != 0)

        [frequencies]
            omega_start         : start of frequency range (imaginary part)
            omega_end           : end of frequency range (imaginary part)
            omega_n             : number of frequencies

        [solver]
            nev                 : number of eigenvalues requested
            shift               : spectral shift (real)
            compute_sensitivity : True/False — compute adjoint modes and wavemaker

    Parameters
    ----------
    filepath : str
        Path to the .ini control file.

    Returns
    -------
    dict with all parameters, ready to use.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError('Control file not found: {0}'.format(filepath))

    cfg = configparser.ConfigParser()
    cfg.read(filepath)

    params = {}

    # [io]
    params['input_path']  = cfg.get('io', 'input_path').strip()
    params['output_path'] = cfg.get('io', 'output_path').strip()

    # [physics]
    params['mach']    = cfg.getfloat('physics', 'mach')
    params['beta']    = cfg.getfloat('physics', 'beta')
    params['nslices'] = cfg.getint('physics', 'nslices')

    # [frequencies] — build listomegas as purely imaginary complex array
    omega_start = cfg.getfloat('frequencies', 'omega_start')
    omega_end   = cfg.getfloat('frequencies', 'omega_end')
    omega_n     = cfg.getint('frequencies',   'omega_n')
    params['listomegas'] = list(np.linspace(omega_start*1j, omega_end*1j, omega_n))

    # [solver]
    params['nev']                 = cfg.getint('solver',   'nev')
    params['shift']               = cfg.getfloat('solver', 'shift')
    params['compute_sensitivity'] = cfg.getboolean('solver', 'compute_sensitivity')

    return params

####################################################################
## RESOLVENT CLASS (DIRECT)
####################################################################

class resolvant(object):
    """Direct resolvent operator: D = P^T M^-1 (L*)^-1 Q L^-1 P"""
    def __init__(self, n, Minv, Qe, J, w, neq):
        self.N = n
        self.Minv = Minv
        self.Q = Qe
        self.P  = None
        self.PT = None
        self.ksp = None
        self.iter = 1
        self.J = J
        self.pcreate(neq)
        pass

    def mult_transpose(self, mat, x, y):
        """ y <- A^H * x """
        print('mult_transpose')
        x.conjugate()
        self.ksp.solveTranspose(x,y)
        y.conjugate()
        pass

    def get_diagonal(self, mat, diag):
        print('get_diag')
        for i in range(self.N):
            diag[i] = self.J[i,i]
        pass

    def mult(self, mat, x, y):
        """ y <- D * x = P^T M^-1 (L*)^-1 Q L^-1 P x """
        Print = PETSc.Sys.Print
        pass
        v1, tmp = self.Q.getVecs()
        v2, tmp = self.Q.getVecs()
        # 1. P * x
        self.P.mult(x, v1)
        # 2. L^-1 * P * x
        self.ksp.solve(v1, v2)
        # 3. Q * L^-1 * P * x
        self.Q.mult(v2, v1)
        # 4. (L*)^-1 * Q * L^-1 * P * x
        v1.conjugate()
        self.ksp.solveTranspose(v1, v2)
        v2.conjugate()
        # 5. M^-1 * (L*)^-1 * Q * L^-1 * P * x
        self.Minv.mult(v2, v1)
        # 6. P^T * M^-1 * (L*)^-1 * Q * L^-1 * P * x
        self.P.multTranspose(v1, y)
        self.iter += 1
        v1.destroy()
        v2.destroy()
        tmp.destroy()
        pass

    def operator(self, w):
        """Build L = iw*I - A and perform LU factorization via MUMPS"""
        Print = PETSc.Sys.Print
        Print(' w = {0}'.format(w))
        self.J.scale(-1.0)
        self.J.shift(w)
        self.ksp = PETSc.KSP().create()
        self.ksp.setOperators(self.J)
        self.ksp.setFromOptions()
        Print('')
        Print(' ******************************************')
        Print('  Performing LU decomposition...')
        Print(' ******************************************')
        Print('')
        self.ksp.setType('preonly')
        pc = self.ksp.getPC()
        pc.setType('lu')
        pc.setFactorSolverType('mumps')
        opts = PETSc.Options("mat_mumps_")
        opts["icntl_7"] = 7
        opts["icntl_8"] = 8
        opts["icntl_4"] = 0
        opts["icntl_11"] = 2
        opts["icntl_33"] = 1
        opts["cntl_3"] = 1e-6
        self.ksp.setUp()
        pass

    def pcreate(self, neq):
        """Build prolongation matrix P: maps velocity DOFs to full (vel+pressure) space"""
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
        Print = PETSc.Sys.Print
        dimrows = self.N
        dimcol = 2*dimrows//neq
        irn = np.zeros(dimcol)
        jcn = np.zeros(dimcol)
        val = np.ones(dimcol)
        for i in range(dimcol):
            jcn[i] = i
        j = 0
        for i in range(0, dimrows, neq):
            irn[j]   = i + 1
            irn[j+1] = i + 2
            j += 2
        ppy  = csr_matrix((val, (irn, jcn)), shape=(dimrows, dimcol))
        ## CREATING PETSc MATRICES
        self.P = PETSc.Mat()
        self.P.create(PETSc.COMM_WORLD)
        self.P.setSizes([dimrows, dimcol])
        self.P.setUp()
        if (size==1):
            self.P.setType('seqaij')
        elif (size>1):
            self.P.setType('mpiaij')
        nnz_row   = ppy.getnnz(axis=1)
        nnz_count = np.concatenate([[0], np.cumsum(nnz_row)])
        RowStart, RowEnd = self.P.getOwnershipRange()
        nrows     = RowEnd - RowStart
        nnz_start = nnz_count[RowStart]
        nnz_end   = nnz_count[RowEnd]
        row_proc  = [(ppy.indptr[RowStart+i] - ppy.indptr[RowStart])
                     for i in range(nrows+1)]
        row_proc[0] = 0
        col_proc  = ppy.indices[nnz_start:nnz_end]
        val_proc  = ppy.data[nnz_start:nnz_end]
        Print(' Assembling P PETSc matrix...')
        self.P.setPreallocationCSR((row_proc, col_proc, val_proc))
        self.P.assemble()
        pass

####################################################################
## RESOLVENT CLASS (ADJOINT)
####################################################################

class resolvant_adjoint(object):
    """Adjoint resolvent operator: D* = P^T M^-1 L^-1 Q (L*)^-1 P
    Reuses the same KSP (LU factorization) as the direct operator.
    D* is obtained by swapping forward and adjoint solves in D.
    """
    def __init__(self, shell_direct):
        self.ksp  = shell_direct.ksp
        self.Q    = shell_direct.Q
        self.Minv = shell_direct.Minv
        self.P    = shell_direct.P
        self.iter = 1

    def mult(self, mat, x, y):
        """y <- D* x = P^T M^-1 L^-1 Q (L*)^-1 P x"""
        pass
        v1, tmp = self.Q.getVecs()
        v2, tmp = self.Q.getVecs()
        # 1. P * x
        self.P.mult(x, v1)
        # 2. (L*)^-1 * P * x  (adjoint solve FIRST — swapped vs. direct)
        v1.conjugate()
        self.ksp.solveTranspose(v1, v2)
        v2.conjugate()
        # 3. Q * (L*)^-1 * P * x
        self.Q.mult(v2, v1)
        # 4. L^-1 * Q * (L*)^-1 * P * x  (forward solve SECOND — swapped vs. direct)
        self.ksp.solve(v1, v2)
        # 5. M^-1 * L^-1 * Q * (L*)^-1 * P * x
        self.Minv.mult(v2, v1)
        # 6. P^T * M^-1 * L^-1 * Q * (L*)^-1 * P * x
        self.P.multTranspose(v1, y)
        self.iter += 1
        v1.destroy()
        v2.destroy()
        tmp.destroy()
        pass

####################################################################
## WAVEMAKER
####################################################################

def compute_wavemaker(direct_vec, adjoint_vec, B, nvars, n, neq,
                      beta, dreduced, rgid, outfile, rank):
    """
    Compute and save the structural sensitivity (wavemaker) field.

    wavemaker_i(x) = |q_adj_i(x)| * |q_dir_i(x)| / |<q_adj_i, B, q_dir_i>|

    Parameters
    ----------
    direct_vec  : PETSc Vec  -- direct mode in full space  (P * f_hat,     size n)
    adjoint_vec : PETSc Vec  -- adjoint mode in full space (P * f_hat_adj, size n)
    B           : PETSc Mat  -- mass matrix (for inner product normalization)
    outfile     : str        -- output .pval filename
    rank        : int        -- MPI rank
    """
    Print = PETSc.Sys.Print

    # B * q_dir
    tmp_vec, _ = B.getVecs()
    B.mult(direct_vec, tmp_vec)

    # Gather all vectors to rank 0
    scatter_d, dir_seq = PETSc.Scatter.toZero(direct_vec)
    scatter_a, adj_seq = PETSc.Scatter.toZero(adjoint_vec)
    scatter_t, tmp_seq = PETSc.Scatter.toZero(tmp_vec)
    scatter_d.scatter(direct_vec,  dir_seq,  False, PETSc.Scatter.Mode.FORWARD)
    scatter_a.scatter(adjoint_vec, adj_seq,  False, PETSc.Scatter.Mode.FORWARD)
    scatter_t.scatter(tmp_vec,     tmp_seq,  False, PETSc.Scatter.Mode.FORWARD)

    if rank == 0:
        dir_arr = dir_seq.getArray()
        adj_arr = adj_seq.getArray()
        tmp_arr = tmp_seq.getArray()

        # Inner product <q_adj | B | q_dir>
        inner = np.dot(np.conj(adj_arr), tmp_arr)
        norm  = abs(inner)
        if norm < 1.0e-30:
            Print(' WARNING: adjoint-direct inner product near zero, wavemaker not normalized')
            norm = 1.0

        # Pointwise wavemaker
        wavemaker   = np.abs(adj_arr) * np.abs(dir_arr) / norm
        wm_vec_np   = wavemaker.astype('c16')
        wm_petsc    = PETSc.Vec().createSeq(len(wm_vec_np))
        wm_petsc.setArray(wm_vec_np)
        mode2pval(outfile, wm_petsc, nvars, n, neq, beta, dreduced, rgid)
        wm_petsc.destroy()

    tmp_vec.destroy()
    pass

####################################################################
## MAIN
####################################################################

def run_slices(params):

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    Print = PETSc.Sys.Print

    # --- Unpack parameters from control file ---
    input_path          = params['input_path']
    output_path         = params['output_path']
    mach                = params['mach']
    beta                = params['beta']
    nslices             = params['nslices']
    listomegas          = params['listomegas']
    nev                 = params['nev']
    shift               = params['shift']
    compute_sensitivity = params['compute_sensitivity']
    rlength             = 1.0
    dreduced            = False

    jacfile = input_path + 'samg.matrix.amg.pval'

    Print('')
    Print(' ========================================')
    Print('  RESOLVENT ANALYSIS — FLORES')
    Print(' ========================================')
    Print(' Input path  : {0}'.format(input_path))
    Print(' Output path : {0}'.format(output_path))
    Print(' Mach        : {0}'.format(mach))
    Print(' beta        : {0}'.format(beta))
    Print(' nev         : {0}'.format(nev))
    Print(' shift       : {0}'.format(shift))
    Print(' Sensitivity : {0}'.format(compute_sensitivity))
    Print(' Frequencies : {0}'.format(listomegas))
    Print(' ========================================')
    Print('')

    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    if (rank==0):
        Print(' Reading Jacobian from {0}'.format(jacfile))
        mjac, neq = openjacobian(jacfile)
        Print('')
        nvars = mjac.shape[0]
        Print(' Matrix main dimension = {0}'.format(nvars))
        Print(' Number of equations   = {0}'.format(neq))
        Print('')

        if (beta==0):
            amatrix = mjac
        else:
            nvars //= nslices
            Print(' J0 block main dimension = {0}'.format(nvars))
            Ly = 1
            Print('Extracting and compacting Jacobian')
            midrow = mjac[nvars*3:nvars*4,:].tocsc()
            jm1 = midrow[:,nvars*2:nvars*3]
            j0  = midrow[:,nvars*3:nvars*4]
            j1  = midrow[:,nvars*4:nvars*5]
            midrow = mjac[nvars*4:nvars*5,:].tocsc()
            njm1 = midrow[:,nvars*3:nvars*4]
            nj0  = midrow[:,nvars*4:nvars*5]
            error = nj0 - j0
            Print(' J0 error = {0}'.format(str(np.max(error.data))))
            error = njm1 - jm1
            Print(' J1 error = {0}'.format(str(np.max(error.data))))
            amatrix = j0 + j1*np.exp(1j*beta*Ly) - j1*np.exp(-1j*beta*Ly)
            Print(' max(A - J0) = {0}'.format(np.max(amatrix-j0)))
            amatrix = amatrix.tocsr()

        n = amatrix.shape[0]
        Print('size of A matrix = {0}'.format(n))
        if (n != nvars):
            sys.exit('N != NVARS... something went very wrong!')
        gridpoints = n//neq

        ## READING MASS MATRIX ##
        Print(' Reading mass matrix and generating M and Inv(M)')
        one = 1. + 0j
        with open(input_path + 'samg.matrix.vol', 'r') as f:
            vols = [float(line) for line in f.readlines()]
        bmatrix  = identity(n, dtype='c16', format='csr')
        mmatinv  = identity(n, dtype='c16', format='csr')
        qematrix = identity(n, dtype='c16', format='csr')
        bmatrix.data[:]  = [vols[i//neq]*one for i in range(n)]
        mmatinv.data[:]  = [1.0/vols[i//neq] for i in range(n)]
        qematrix.data[:] = [vols[i//neq]*one for i in range(n)]

        ###### DOMAIN REDUCTION #####
        if dreduced:
            Print(' Applying domain reduction')
            coord = read_coordinates('coord.dat', rlength, beta)
            dr = domain_reduction(0, 0.5, -0.4, 1.0)
            dr.create_Pmatrix(coord)
            amatrix  = dr.reduce_matrix(amatrix)
            bmatrix  = dr.reduce_matrix(bmatrix)
            qematrix = dr.reduce_matrix(qematrix)
            mmatinv  = dr.reduce_matrix(mmatinv)
            n = amatrix.shape[0]
            localid = np.repeat(np.arange(0, gridpoints, 1, dtype='i4'), neq)
            rgid = dr.reduce_vector(localid)[0::neq].astype(int)
        else:
            rgid = None

    ## END IF (RANK==0)

    # Broadcast
    if rank != 0:
        n = None; neq = None
    n   = comm.bcast(n,   root=0)
    neq = comm.bcast(neq, root=0)
    if rank != 0:
        amatrix  = csr_matrix((n,n), dtype='c16')
        bmatrix  = csr_matrix((n,n), dtype='c16')
        mmatinv  = csr_matrix((n,n), dtype='c16')
        qematrix = csr_matrix((n,n), dtype='c16')
    amatrix  = comm.bcast(amatrix,  root=0)
    bmatrix  = comm.bcast(bmatrix,  root=0)
    mmatinv  = comm.bcast(mmatinv,  root=0)
    qematrix = comm.bcast(qematrix, root=0)

    ## CREATING PETSc MATRICES
    def make_petsc_mat(scipy_mat, label):
        M = PETSc.Mat(); M.create(PETSc.COMM_WORLD)
        M.setSizes([n, n]); M.setUp()
        M.setType('seqaij' if size==1 else 'mpiaij')
        nnz_row   = scipy_mat.getnnz(axis=1)
        nnz_count = np.concatenate([[0], np.cumsum(nnz_row)])
        RS, RE    = M.getOwnershipRange()
        nr        = RE - RS
        ns        = nnz_count[RS]; ne = nnz_count[RE]
        rp = [(scipy_mat.indptr[RS+i] - scipy_mat.indptr[RS]) for i in range(nr+1)]
        rp[0] = 0
        M.setPreallocationCSR((rp, scipy_mat.indices[ns:ne], scipy_mat.data[ns:ne]))
        M.assemble()
        Print(' Assembled {0}'.format(label))
        return M

    A    = make_petsc_mat(amatrix,  'A (Jacobian)')
    fac  = 1. / (mach * np.sqrt(1.4))
    A.scale(fac)

    # Diagonal matrices assembled entry by entry (faster for diagonal)
    def make_diag_petsc(data_vec, label):
        M = PETSc.Mat(); M.create(PETSc.COMM_WORLD)
        M.setSizes([n, n]); M.setUp()
        M.setType('seqaij' if size==1 else 'mpiaij')
        RS, RE = M.getOwnershipRange()
        for pt in range(RS, RE):
            M[pt, pt] = data_vec[pt]
        M.assemble()
        Print(' Assembled {0}'.format(label))
        return M

    B    = make_diag_petsc(bmatrix.data,  'B (mass)')
    Binv = make_diag_petsc(mmatinv.data,  'Binv (mass inverse)')
    Q    = make_petsc_mat(qematrix, 'Q (energy weight)')

    amatrix = None; bmatrix = None; mmatinv = None; qematrix = None

    #####################################################
    ###########      EIGENVALUE SOLVER      #############
    #####################################################
    ncv = nev*3 + 1
    mpd = ncv - 1

    for omega in listomegas:

        Print('\n *** omega = {0} ***\n'.format(omega))

        # --------------------------------------------------
        # 1. Build shell and LU factorization (shared by direct + adjoint)
        # --------------------------------------------------
        R = PETSc.Mat().create()
        R.setSizes([n//2, n//2])
        R.setType('python')
        shell = resolvant(n=n, Minv=Binv, Qe=Q, J=A, w=omega, neq=neq)
        R.setPythonContext(shell)
        R.setUp()
        shell.operator(omega)  # <-- single LU factorization, reused everywhere

        # --------------------------------------------------
        # 2. Direct problem
        # --------------------------------------------------
        E = SLEPc.EPS().create()
        E.setOperators(R)
        E.setProblemType(SLEPc.EPS.ProblemType.NHEP)
        E.setType('krylovschur')
        E.setTolerances(tol=1e-6, max_it=1000)
        E.setDimensions(nev, ncv, mpd)
        if shift != 0.0:
            ST = E.getST()
            ST.setShift(shift)
        E.setFromOptions()

        if rank==0:
            print(' Target = ', E.getTarget())
            print(' Omega  = ', omega)
            print(' nev    = ', nev, '  ncv = ', ncv)

        Print(' SOLVING DIRECT PROBLEM\n')
        E.solve()

        nconv = E.getConverged()
        if rank==0:
            print(' Converged direct eigenvalues: ', nconv)

        xr,      tmp = shell.P.getVecs()
        xi,      tmp = shell.P.getVecs()
        eigpre,  tmp = A.getVecs()   # P * f_hat      (full space)
        resppre, tmp = A.getVecs()   # L^-1 * P * f_hat (full space)
        eigs         = []
        direct_modes = []            # stored for wavemaker

        if nconv > 0:
            Print("           k           ||Ax-kx||/||kx|| ")
            Print("--------------------- ------------------")
            for i in range(nconv):
                k = E.getEigenpair(i, xr, xi)
                eigs.append(k)
                error = E.computeError(i)
                if k.imag != 0.0:
                    Print(" %9f%+9f j    %12g" % (k.real, k.imag, error))
                else:
                    Print(" %12f         %12g" % (k.real, error))

                # Save optimal forcing
                eigvecfile = output_path+'/eigf_'+str(i)+'_'+str(omega)+'.pval'
                shell.P.mult(xr, eigpre)
                scatter, eigenvec = PETSc.Scatter.toZero(eigpre)
                scatter.scatter(eigpre, eigenvec, False, PETSc.Scatter.Mode.FORWARD)
                if rank==0:
                    mode2pval(eigvecfile, eigenvec, nvars, n, neq, beta, dreduced, rgid)

                # Save optimal response
                shell.ksp.solve(eigpre, resppre)
                respvecfile = output_path+'/eigr_'+str(i)+'_'+str(omega)+'.pval'
                scatter_r, respvec = PETSc.Scatter.toZero(resppre)
                scatter_r.scatter(resppre, respvec, False, PETSc.Scatter.Mode.FORWARD)
                if rank==0:
                    mode2pval(respvecfile, respvec, nvars, n, neq, beta, dreduced, rgid)

                # Store copy for wavemaker
                if compute_sensitivity:
                    mode_copy, _ = A.getVecs()
                    eigpre.copy(mode_copy)
                    direct_modes.append(mode_copy)

        eigpre.destroy(); resppre.destroy(); xi.destroy(); xr.destroy()
        E.destroy()

        if rank==0:
            eigv_file = output_path+'/eigv_DIR_'+str(omega)+'.dat'
            with open(eigv_file, 'w') as w:
                for i in range(nconv):
                    w.write('{0:2d}   {1:12.8f}   {2:12.8f}\n'.format(
                        i, eigs[i].real, eigs[i].imag))
            print(' Saved eigenvalues -> ', eigv_file)

        # --------------------------------------------------
        # 3. Adjoint problem (reuses LU — no extra factorization)
        # --------------------------------------------------
        if compute_sensitivity and nconv > 0:

            Print(' SOLVING ADJOINT PROBLEM (reusing LU)\n')

            R_adj = PETSc.Mat().create()
            R_adj.setSizes([n//2, n//2])
            R_adj.setType('python')
            shell_adj = resolvant_adjoint(shell)
            R_adj.setPythonContext(shell_adj)
            R_adj.setUp()

            E_adj = SLEPc.EPS().create()
            E_adj.setOperators(R_adj)
            E_adj.setProblemType(SLEPc.EPS.ProblemType.NHEP)
            E_adj.setType('krylovschur')
            E_adj.setTolerances(tol=1e-6, max_it=1000)
            E_adj.setDimensions(nev, ncv, mpd)
            E_adj.setFromOptions()
            E_adj.solve()

            nconv_adj = E_adj.getConverged()
            if rank==0:
                print(' Converged adjoint eigenvalues: ', nconv_adj)

            xr_adj,      tmp = shell.P.getVecs()
            xi_adj,      tmp = shell.P.getVecs()
            eigpre_adj,  tmp = A.getVecs()
            eigs_adj         = []

            if nconv_adj > 0:
                Print(" ADJOINT MODES:")
                Print("           k*          ||Ax-kx||/||kx|| ")
                Print("--------------------- ------------------")
                for i in range(nconv_adj):
                    k_adj = E_adj.getEigenpair(i, xr_adj, xi_adj)
                    eigs_adj.append(k_adj)
                    error_adj = E_adj.computeError(i)
                    if k_adj.imag != 0.0:
                        Print(" %9f%+9f j    %12g" % (k_adj.real, k_adj.imag, error_adj))
                    else:
                        Print(" %12f         %12g" % (k_adj.real, error_adj))

                    # Save adjoint mode
                    adjvecfile = output_path+'/eiga_'+str(i)+'_'+str(omega)+'.pval'
                    shell.P.mult(xr_adj, eigpre_adj)
                    scatter_a, adjvec = PETSc.Scatter.toZero(eigpre_adj)
                    scatter_a.scatter(eigpre_adj, adjvec, False, PETSc.Scatter.Mode.FORWARD)
                    if rank==0:
                        mode2pval(adjvecfile, adjvec, nvars, n, neq, beta, dreduced, rgid)

                    # Compute and save wavemaker
                    if i < len(direct_modes):
                        wmfile = output_path+'/wavemaker_'+str(i)+'_'+str(omega)+'.pval'
                        compute_wavemaker(direct_modes[i], eigpre_adj, B,
                                          nvars, n, neq, beta, dreduced, rgid,
                                          wmfile, rank)
                        if rank==0:
                            print(' Saved wavemaker mode {0} -> {1}'.format(i, wmfile))

            if rank==0:
                eigv_adj_file = output_path+'/eigv_ADJ_'+str(omega)+'.dat'
                with open(eigv_adj_file, 'w') as w:
                    for i in range(nconv_adj):
                        w.write('{0:2d}   {1:12.8f}   {2:12.8f}\n'.format(
                            i, eigs_adj[i].real, eigs_adj[i].imag))
                print(' Saved adjoint eigenvalues -> ', eigv_adj_file)

            eigpre_adj.destroy(); xr_adj.destroy(); xi_adj.destroy()
            R_adj.destroy(); E_adj.destroy()

        # --------------------------------------------------
        # 4. Cleanup
        # --------------------------------------------------
        for v in direct_modes:
            v.destroy()
        R.destroy()

    # end for omega


####################################################################

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print('Usage: python RESOLVANT.py <control_file.ini>')
        print('       mpirun -n 4 python RESOLVANT.py <control_file.ini>')
        sys.exit(1)

    ctrl_file = sys.argv[1]
    params    = read_control_file(ctrl_file)
    run_slices(params)