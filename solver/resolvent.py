#! /usr/bin/env python

import numpy as np
# import matplotlib.pyplot as plt
import sys,os
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

class resolvant(object):
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
#        self.operator(J,w)
        pass

    def mult_transpose(self, mat, x, y):
        """ y <- A^H * x """
        print ('mult_transpose')
        x.conjugate()
        self.ksp.solveTranspose(x,y)
        y.conjugate()
        pass
        
    def get_diagonal(self, mat, diag):
        print ('get_diag')
        for i in range(n):
            diag[i] = J[i,i]
        pass

    def mult(self, mat, x, y):
        """ y <- A * x """
#        print ('mult')
        print = PETSc.Sys.print
        
#        self.ksp.solve(x,y)
        pass
        
        v1, tmp = self.Q.getVecs()
        v2, tmp = self.Q.getVecs()
        # 1. P * fs
        self.P.mult(x, v1)

        # 2. C^-1 * P * fs
        self.ksp.solve(v1,v2)

        # 3. Q * C^-1 * P * fs
        self.Q.mult(v2,v1)
        # v2.copy(v1)

        # 4. (C*)^-1 * Q * C^-1 * P * fs
        v1.conjugate()
        self.ksp.solveTranspose(v1,v2)
        v2.conjugate()
        # v2.copy(y)

        # 5. M^-1 * (C*)^-1 * Q * C^-1 * P * fs
        self.Minv.mult(v2,v1)
        # v2.copy(v1)

        # 6. PT * M^-1 * (C*)^-1 * Q * C^-1 * P * fs
        self.P.multTranspose(v1, y)
        # print( ' Succesfull calls to MatMul = {0}'.format(self.iter) )
        self.iter += 1
        v1.destroy()
        v2.destroy()
        tmp.destroy()
        pass

    def operator(self, w):
        """
        ###################
        ## CREATE THE KSP OBJECT FOR THE RESOLVANT OPERATOR
        ###################
        """
        print = PETSc.Sys.print
        print(' w = {0}'.format(w))
        self.J.scale(-1.0)
        self.J.shift(w)
        self.ksp = PETSc.KSP().create()
        self.ksp.setOperators(self.J)
        self.ksp.setFromOptions()
        tol = 1e-7
        print('')
        print(' ******************************************')
        print('  Performing LU decomposition...')
        print(' ******************************************')
        print('')
        ## LU options ##
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
        """
        Generates de prolongation matrices P and PT
        """
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
        print = PETSc.Sys.print
        dimrows = self.N
        dimcol = 2*dimrows//neq

        irn = np.zeros(dimcol)
        jcn = np.zeros(dimcol)
        val = np.ones(dimcol)

        for i in range(dimcol):
          jcn[i] = i
        j = 0
        for i in range(0,dimrows,neq):
          irn[j] = i + 1
          irn[j+1] = i + 2
          j +=2
        ppy = csr_matrix((val, (irn, jcn)), shape=(dimrows, dimcol))
        ppyt = ppy.transpose()

        ## CREATING PETSc MATRICES
        self.P = PETSc.Mat()
        self.P.create(PETSc.COMM_WORLD)
        self.P.setSizes([dimrows, dimcol])
        self.P.setUp()
        if (size==1):
            self.P.setType('seqaij')
        elif (size>1):
            self.P.setType('mpiaij')

        ## Paralelizing algorithm
        nnz_row = ppy.getnnz(axis=1)
        nnz_count = np.concatenate([[0],np.cumsum(nnz_row)])
        RowStart, RowEnd = self.P.getOwnershipRange()
        nrows = RowEnd - RowStart
        nnz_start = nnz_count[RowStart]
        nnz_end = nnz_count[RowEnd]
        row_proc = [(ppy.indptr[RowStart+i] - ppy.indptr[RowStart])
                        for i in range(nrows+1)]
        row_proc[0] = 0
        nnzproc = 0
        nnzproc = nnz_end - nnz_start
        col_proc = ppy.indices[nnz_start:nnz_end]
        val_proc = ppy.data[nnz_start:nnz_end]
        print(' Assembling PETSc matrix...')
        self.P.setPreallocationCSR((row_proc, col_proc, val_proc))
        self.P.assemble()
        # self.PT = self.P.duplicate(copy=True)
        # self.PT.transpose()
        pass

####################################################################

def run_slices():
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    print = PETSc.Sys.print


    # Path to the Jacobian and volumes files (if gen = False, the volumes file is not needed, as the Jacobian is already scaled by the cell volumes)
    input_file_path = '/home/w059/w059589/projects/07_TRANSDIFFUSE/02_BFS/JAC/'

    #######################
    jacfile = input_file_path + 'samg.matrix.amg.pval'
    qemfile = input_file_path + 'qematrix.pval'
    #######################
    beta = 0 
    nslices = 7
    omega = 0.1j
    #######################
    mach = 0.1 # Mach number
    rlength = 1.0
    #######################
    dreduced = False # Reduce the domain to a box defined by xmin/xmax and zmin/zmax
    xmin = -0.4
    xmax = 1
    zmin = 0
    zmax = 0.5
    #######################
    nev = 30
    adjoint = False
    the_shift = 0
    #######################

    if ( os.path.isdir('./RESULTS_resolvent') ):
        pass
    else:
        os.mkdir('./RESULTS_resolvent')

    if (rank==0):
        print('')
        print(' Reading Jacobian from {0}'.format(jacfile))
        mjac, neq = openjacobian(jacfile)
        print(' Reading Qe Matrix from {0}'.format(qemfile))
        print('')
        # qematrix, foo = openqe(qemfile)
        nvars = mjac.shape[0]
        print(' Matrix main dimension = {0}'.format(nvars) )
        print(' Number of equations = {0}'.format(neq))
        print('')

        if (beta==0):
            amatrix = mjac
        else:
            nvars //= nslices
            print(' J0 block main dimension = {0}'.format(nvars) )
            Ly = 1

            print('Extracting and compacting Jacobian')
            midrow = mjac[nvars*3:nvars*4,:]
            midrow = midrow.tocsc()

            jm1 = midrow[:,nvars*2:nvars*3]
            j0  = midrow[:,nvars*3:nvars*4]
            j1  = midrow[:,nvars*4:nvars*5]

            midrow = mjac[nvars*4:nvars*5,:]
            midrow = midrow.tocsc()

            njm1 = midrow[:,nvars*3:nvars*4]
            nj0  = midrow[:,nvars*4:nvars*5]
            nj1  = midrow[:,nvars*4:nvars*5]

            error = nj0 - j0
            print(' J0 error = {0}'.format( str( np.max(error.data) ) ) )

            error = njm1 - jm1
            print(' J1 error = {0}'.format( str( np.max(error.data) ) ) )
            # sys.exit()
            amatrix = j0 + j1*np.exp(1j*beta*Ly) - j1*np.exp(-1j*beta*Ly)

            print(' max(A - J0) = {0}'.format( np.max(amatrix-j0) ) )

            amatrix = amatrix.tocsr()

        n = amatrix.shape[0]
        
        print('size of A matrix = {0}'.format(n))
        if (n!=nvars):
            sys.exit('N != NVARS... something went very wrong!')
        gridpoints = n//neq

        ## READING MASS MATRIX AND STORE IT IN SPARSE FORMAT ##
        print(' Reading mass matrix and generating M and Inv(M)')
        print('')
        one = 1. + 0j
        with open(input_file_path + 'samg.matrix.vol', 'r') as f:
            vols = f.readlines()
            vols = [float(line) for line in vols]
        bmatrix = identity(n, dtype='c16', format='csr')  # Matrix M
        mmatinv = identity(n, dtype='c16', format='csr')  # Matrix Inv(M)
        qematrix = identity(n, dtype='c16', format='csr')  # Matrix Qe (same volumes as B)
        bmatrix.data[:] = [vols[i//neq]*one for i in range(n)]
        mmatinv.data[:] = [1.0/vols[i//neq] for i in range(n)]
        qematrix.data[:] = [vols[i//neq]*one for i in range(n)]
        


        ###### DOMAIN REDUCTION #####
        if (dreduced==True):
            print('')
            print(' Applying domain reduction')
            print(' XMIN/XMAX = {0}/{1}'.format(xmin,xmax))
            print(' ZMIN/ZMAX = {0}/{1}'.format(zmin,zmax))
            # 1. Read coordinates from file
            coord = read_coordinates('coord.dat', rlength, beta)
            # 2. Generate the permutation matrix
            dr = domain_reduction(zmin, zmax, xmin, xmax)
            dr.create_Pmatrix(coord)
            # 3. Reorder and slice the jacobian and volumes matrix
            print(' Previous number of NNZ elem = {0}'.format(amatrix.nnz))
            amatrix = dr.reduce_matrix(amatrix)
            print(' New number of NNZ elem = {0}'.format(amatrix.nnz))
            bmatrix = dr.reduce_matrix(bmatrix)
            qematrix = dr.reduce_matrix(qematrix)
            mmatinv = dr.reduce_matrix(mmatinv)
            n = amatrix.shape[0]
            print(' New leading dimension of A Matrix = {0}'.format(n))
            # 5. Generate a local_id vector, reorder and slice it
            #    (this will be needed for output indexing)
            localid = np.arange(0, gridpoints, 1, dtype='i4')
            localid = np.repeat(localid, neq)
            rgid = dr.reduce_vector(localid)
            rgid = rgid[0::neq].astype(int)
            print('')
        else:
            rgid = None


    ## END IF (RANK==0)

    # Broadcast variables
    if rank!=0:
        n   = None
        neq = None
    n   = comm.bcast(n, root=0)
    neq = comm.bcast(neq, root=0)
    if rank!=0:
        amatrix  = csr_matrix((n,n), dtype='c16')
        bmatrix  = csr_matrix((n,n), dtype='c16')
        mmatinv  = csr_matrix((n,n), dtype='c16')
        qematrix = csr_matrix((n,n), dtype='c16')
    amatrix  = comm.bcast(amatrix, root=0)
    bmatrix  = comm.bcast(bmatrix, root=0)
    mmatinv  = comm.bcast(mmatinv, root=0)
    qematrix = comm.bcast(qematrix, root=0)

    ## CREATING PETSc MATRICES
    #################### A matrix (JACOBIAN) #############################
    A = PETSc.Mat()
    A.create(PETSc.COMM_WORLD)
    A.setSizes([n, n])
    A.setUp()
    if (size==1):
        A.setType('seqaij')
    elif (size>1):
        A.setType('mpiaij')

    ## Paralelizing algorithm
    nnz_row = amatrix.getnnz(axis=1)
    nnz_count = np.concatenate([[0],np.cumsum(nnz_row)])
    RowStart, RowEnd = A.getOwnershipRange()
    nrows = RowEnd - RowStart
    nnz_start = nnz_count[RowStart]
    nnz_end = nnz_count[RowEnd]
    row_proc = [(amatrix.indptr[RowStart+i] - amatrix.indptr[RowStart])
                    for i in range(nrows+1)]
    row_proc[0] = 0
    nnzproc = 0
    nnzproc = nnz_end - nnz_start
    col_proc = amatrix.indices[nnz_start:nnz_end]
    val_proc = amatrix.data[nnz_start:nnz_end]
    print(' Assembling A PETSc matrix...')
    A.setPreallocationCSR((row_proc, col_proc, val_proc))
    A.assemble()
    ## TAU Matrix Scaling
    fac = 1. / (mach*np.sqrt(1.4))
    A.scale(fac)

    # Freeing memory
    col_proc = None
    val_proc = None
    row_proc = None

    #################### MASS MATRIX ######################
    B = PETSc.Mat()
    B.create(PETSc.COMM_WORLD)
    B.setSizes([n, n])
    B.setUp()
    if (size==1):
        B.setType('seqaij')
    elif (size>1):
        B.setType('mpiaij')
    RowStart, RowEnd = B.getOwnershipRange()
    for pt in range(RowStart, RowEnd):
        B[pt,pt] = bmatrix.data[pt]
    print(' Assembling B PETSc matrix...')
    B.assemble()

    Binv = PETSc.Mat()
    Binv.create(PETSc.COMM_WORLD)
    Binv.setSizes([n, n])
    Binv.setUp()
    if (size==1):
        Binv.setType('seqaij')
    elif (size>1):
        Binv.setType('mpiaij')
    RowStart, RowEnd = Binv.getOwnershipRange()
    for pt in range(RowStart, RowEnd):
        Binv[pt,pt] = mmatinv.data[pt]
    print(' Assembling Binv PETSc matrix...')
    Binv.assemble()

    #################### QE MATRIX ######################
    Q = PETSc.Mat()
    Q.create(PETSc.COMM_WORLD)
    Q.setSizes([n, n])
    Q.setUp()
    if (size==1):
        Q.setType('seqaij')
    elif (size>1):
        Q.setType('mpiaij')
#    RowStart, RowEnd = Q.getOwnershipRange()
#    for pt in range(RowStart, RowEnd):
#        Q[pt,pt] = qematrix.data[pt]
#    print(' Assembling Qe PETSc matrix...')
#    Q.assemble()

#    ## Paralelizing algorithm
    nnz_row = qematrix.getnnz(axis=1)
    nnz_count = np.concatenate([[0],np.cumsum(nnz_row)])
    RowStart, RowEnd = Q.getOwnershipRange()
    nrows = RowEnd - RowStart
    nnz_start = nnz_count[RowStart]
    nnz_end = nnz_count[RowEnd]
    row_proc = [(qematrix.indptr[RowStart+i] - qematrix.indptr[RowStart])
                    for i in range(nrows+1)]
    row_proc[0] = 0
    nnzproc = 0
    nnzproc = nnz_end - nnz_start
    col_proc = qematrix.indices[nnz_start:nnz_end]
    val_proc = qematrix.data[nnz_start:nnz_end]
    print(' Assembling Q PETSc matrix...')
    Q.setPreallocationCSR((row_proc, col_proc, val_proc))
    Q.assemble()

    # Freeing memory
    col_proc = None
    val_proc = None
    row_proc = None

    amatrix  = None
    bmatrix  = None
    mmatinv  = None
    qematrix = None

    #####################################################
    ###########      EIGENVALUE SOLVER      #############
    #####################################################
    ncv = nev*3 + 1
    mpd = ncv-1
    listomegas = [57.2j] #,60j,70j,80j,90j,100j] #np.linspace(0.005, 100, 10)
    print(listomegas)
    for omega in listomegas:
        # setup linear system matrix
        R = PETSc.Mat().create()
        R.setSizes([n//2, n//2])
#        R.setSizes([n, n])
        R.setType('python')
        shell = resolvant(n=n, Minv=Binv, Qe=Q, J=A, w=omega, neq=neq) # shell context
        R.setPythonContext(shell)
        R.setUp()
        # Setup the eigensolver
        E = SLEPc.EPS().create()
        E.setOperators(R)
        E.setProblemType( SLEPc.EPS.ProblemType.NHEP )

        ###################################
        ## Eigenvalue Solver Options
        ###################################
        tname = 'krylovschur'
        E.setType(tname)
        E.setTolerances(tol=1e-6, max_it=1000)
        E.setDimensions(nev, ncv, mpd)
#        E.setWhichEigenpairs(E.Which.TARGET_MAGNITUDE)
#        E.setTarget(0.0)
        E.setFromOptions()

        if (rank==0):
            print(' ')
            print(' Target = ', E.getTarget())
            print(' Resolvant OMEGA = ', omega)
            print(' EGV Solver = ', E.getType())
            print(' Region of Spectrum = ', E.getWhichEigenpairs())
            print(' Number of eigenvalues requested = ', nev)
            print(' Number of column vectors = ', ncv)
            print(' Maximum dimension of projected problem = ', mpd)
            print('')
        #------------------#
        shell.operator(omega)
        print(' SOLVING!\n')
        E.solve()
        #------------------#
        #######################################
        ## POST-PROCESSING
        #######################################
        its = E.getIterationNumber()
        tol, max_it = E.getTolerances()
        ksp_its = shell.iter
        nconv = E.getConverged()

        if (rank==0):
            print ('')
            print(' Number of iterations of the EPS method: ', its)
            print(' CRITERIA: tol= ', tol,', maxit= ',max_it)
            print(' Number of iterations of the ksp method: ', ksp_its)
            print(' Number of requested eigenvalues: ', nev)
            print(' Number of converged eigenvalues: ', nconv)

        xr, tmp = shell.P.getVecs()
        xi, tmp = shell.P.getVecs()
        eigpre, tmp = A.getVecs()
        eigs = []
        if nconv > 0:
            print("")
            print("           k           ||Ax-kx||/||kx|| ")
            print("--------------------- ------------------")
            for i in range(nconv):
                k = E.getEigenpair(i, xr, xi)
                eigs.append(k)
                error = E.computeError(i)
                if k.imag != 0.0:
                    print(" %9f%+9f j    %12g" % (k.real, k.imag, error))
                else:
                    print(" %12f         %12g" % (k.real, error))

                eigvecfile = './RESULTS_resolvent/eigf_'+str(i)+'.pval'
                shell.P.mult(xr,eigpre)
                scatter, eigenvec = PETSc.Scatter.toZero(eigpre)
                scatter.scatter(eigpre, eigenvec, False, PETSc.Scatter.Mode.FORWARD)
                if (rank==0):
                    mode2pval(eigvecfile, eigenvec, nvars, n, neq, beta, dreduced, rgid)


            print("")

        eigpre.destroy()
        xi.destroy()
        xr.destroy()
        R.destroy()
        E.destroy()

        if (rank==0):
            if(adjoint):
                eigv_file = './RESULTS_resolvent/eigv_ADJ.dat'
            else:
                eigv_file = './RESULTS_resolvent/eigv_DIR_'+str(omega)+'.dat'

            print(' Saving Eigenvalues in ' + eigv_file)
            with open(eigv_file, 'w') as w:
                for i in range(nconv):
                    w.write('{0:2d}   {1:12.8f}   {2:12.8f}\n'.format(i,
                                                                    eigs[i].real,
                                                                    eigs[i].imag))
            print(' DONE')
            print('')


if __name__ == "__main__":
    run_slices()
