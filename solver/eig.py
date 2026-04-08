#! /usr/bin/env python

import numpy as np
# import matplotlib.pyplot as plt
import sys,os
from netCDF4 import Dataset
from scipy.sparse import csr_matrix, csc_matrix, linalg as sla
from scipy.sparse import identity


from jac_red import domain_reduction
from save2pval import mode2pval, mode2pval3D
from input_output import openjacobian, read_coordinates, openegvec, opendualgrid
import pdb

import petsc4py
import slepc4py
slepc4py.init(sys.argv)
from petsc4py import PETSc
from slepc4py import SLEPc
from mpi4py import MPI

def run_slices():
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    Print = PETSc.Sys.Print

    #######################
    jacfile = './DATA/samg.matrix.amg.pval'
    volfile = './DATA/samg.matrix.vol'
    #######################
    beta = 0.0
    nslices = 7
    #######################
    mach = 0.34
    rlength = 0.16
    #######################
    dreduced = True
    xmin = 0.02
    xmax = 0.16
    zmin = -0.03
    zmax = 0.03
    #######################
    nev = 20
    gen = False
    idordered = True
    adjoint = False
    the_shift = -0.1 + 5.j
    #######################
    
    fac = 1. / (mach*np.sqrt(1.4))

    if ( os.path.isdir('./RESULTS') ):
        pass
    else:
        os.mkdir('./RESULTS')

    if (rank==0):
        Print('')
        Print(' Reading Jacobian')
        mjac, neq = openjacobian(jacfile)
#        mjac.data *= fac
        nvars = mjac.shape[0]
        Print(' Matrix main dimension = {0}'.format(nvars) )
        Print(' Number of equations = {0}'.format(neq))
        Print('')

        if (beta==0):
            amatrix = mjac
        else:
            nvars /= nslices
            Print(' J0 block main dimension = {0}'.format(nvars) )
            Ly = 1.0 # 2*np.pi
            beta = 0.9 #2*np.pi*beta
            #0.2991993
            #62.83185307
            rj = np.exp(1j*beta*Ly)
            r2j = np.exp(2*1j*beta*Ly)

            Print('Extracting and compacting Jacobian')
            midrow = mjac[nvars*3:nvars*4,:]
            midrow = midrow.tocsc()

            jm2 = midrow[:,nvars:nvars*2]
            jm1 = midrow[:,nvars*2:nvars*3]
            j0  = midrow[:,nvars*3:nvars*4]
            j1  = midrow[:,nvars*4:nvars*5]
            j2  = midrow[:,nvars*5:nvars*6]

            midrow = mjac[nvars*4:nvars*5,:]
            midrow = midrow.tocsc()

            njm1 = midrow[:,nvars*3:nvars*4]
            nj0  = midrow[:,nvars*4:nvars*5]
            nj1  = midrow[:,nvars*4:nvars*5]

            error = nj0 - j0
            Print(' J0 error = {0}'.format( str( np.max(error.data) ) ) )

            error = njm1 - jm1
            Print(' J1 error = {0}'.format( str( np.max(error.data) ) ) )

            amatrix = j0 + j1*rj + jm1*1./rj + j2*r2j + jm2*1./r2j

#            amatrix = j0 + j1 + jm1 + jm2 + j2

            Print (' max(A - J0) = {0}'.format( np.max(amatrix-j0) ) )

            amatrix = amatrix.tocsr()

        n = amatrix.shape[0]
        if (n!=nvars):
            sys.exit('N != NVARS... something went very wrong!')
        gridpoints = n/neq

        ## READING MASS MATRIX AND STORE IT IN SPARSE FORMAT ##
        Print (' Reading mass matrix and generating M and Inv(M)')
        Print ('')
        one = 1. + 0j
        with open(volfile, 'r') as f:
            vols = f.readlines()
            vols = [float(line) for line in vols]
        bmatrix    = identity(n, dtype='c16', format='csr')  # Matrix M
        mmatinv = identity(n, dtype='c16', format='csr')  # Matrix Inv(M)
        bmatrix.data[:] = [vols[i//neq]*one for i in xrange(n)]
        mmatinv.data[:] = [1./(vols[i//neq]*one) for i in xrange(n)]
        # print bmatrix[0:80]
#        bmatrix.data[:] = [1.0/vols[i/neq]*one for i in xrange(n)]

        # Localid for opening/writing
        if (idordered==True):
            local_id = np.arange(0, gridpoints, 1, dtype='i4')
        else:
            local_id = opendualgrid('./DATA/dualgrid2D')

        ###### DOMAIN REDUCTION #####
        if (dreduced==True):
            Print('')
            Print(' Applying domain reduction')
            Print(' XMIN/XMAX = {0}/{1}'.format(xmin,xmax))
            Print(' ZMIN/ZMAX = {0}/{1}'.format(zmin,zmax))
            # 1. Read coordinates from file
            coord = read_coordinates('./DATA/samg.matrix.coo', rlength, beta)
            # 2. Generate the permutation matrix
            dr = domain_reduction(zmin, zmax, xmin, xmax)
            dr.create_Pmatrix(coord)
            # 3. Reorder and slice the jacobian and volumes matrix
            Print(' Previous number of NNZ elem = {0}'.format(amatrix.nnz))
            amatrix = dr.reduce_matrix(amatrix)
            Print(' New number of NNZ elem = {0}'.format(amatrix.nnz))
            bmatrix = dr.reduce_matrix(bmatrix)
            mmatinv = dr.reduce_matrix(mmatinv)
            n = amatrix.shape[0]
            Print(' New leading dimension of A Matrix = {0}'.format(n))
            # 5. Generate a local_id vector, reorder and slice it
            #    (this will be needed for output indexing)
            local_id = np.repeat(local_id, neq)
            rgid = dr.reduce_vector(local_id)
            rgid = rgid[0::neq].astype(int)
            Print('')
        else:
            rgid = None

    ## END IF (RANK==0)

    # Broadcast variables
    if rank!=0:
        n = None
    n = comm.bcast(n, root=0)
    if rank!=0:
        amatrix = csr_matrix((n,n), dtype='c16')
        bmatrix = csr_matrix((n,n), dtype='c16')
    amatrix = comm.bcast(amatrix, root=0)
    bmatrix = comm.bcast(bmatrix, root=0)

    ## CREATING PETSc MATRICES
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
                    for i in xrange(nrows+1)]
    row_proc[0] = 0
    nnzproc = 0
    nnzproc = nnz_end - nnz_start
    col_proc = amatrix.indices[nnz_start:nnz_end]
    val_proc = amatrix.data[nnz_start:nnz_end]
    Print (' Assembling PETSc matrix...')
    A.setPreallocationCSR((row_proc, col_proc, val_proc))
    A.assemble()
    ## TAU Matrix Scaling
    A.scale(fac)

    # Freeing memory
    col_proc = None
    val_proc = None
    row_proc = None

    B = PETSc.Mat()
    B.create(PETSc.COMM_WORLD)
    B.setSizes([n, n])
    B.setUp()
    if (size==1):
        B.setType('seqaij')
    elif (size>1):
        B.setType('mpiaij')
    RowStart, RowEnd = B.getOwnershipRange()
    for pt in xrange(RowStart, RowEnd):
        B[pt,pt] = bmatrix.data[pt]
    B.assemble()

    ######################################################
    ######################################################
#    C = PETSc.Mat()
#    C = A.duplicate()
#    B.matMult(A,C)
#    B.destroy()
#    A.destroy()
    Print('\n################################')
    if (adjoint):
        A.transpose()
        Print('    SOLVING ADJOINT PROBLEM')
    else:
        Print('    SOLVING DIRECT PROBLEM')
    ##
    Print('################################\n')
    ######################################################
    ######################################################

    ncv = nev*3 + 1
    mpd = ncv-1

    # Setup the eigensolver
    E = SLEPc.EPS().create()
    if (gen):
        Print('  Generalised EGVP  Ax=sMx')
        E.setOperators(A,B)
        E.setProblemType( SLEPc.EPS.ProblemType.GNHEP )
    else:
        Print('  Standard EGVP   Ax=sx')
        E.setOperators(A)
        E.setProblemType( SLEPc.EPS.ProblemType.NHEP )

    #############################
    ## Spectral Tranformation
    #############################
    ST = E.getST()
    st_type = 'sinvert'
    ST.setType(st_type)
    ST.setShift(the_shift)
    ST.setFromOptions()

    ###############################
    ## Preconditioner (LU) options
    ###############################
    K = ST.getKSP()
    ksp_type = 'preonly'
    K.setType(ksp_type)
    K.setFromOptions()
    pc = K.getPC()
    pc.setType('lu')
    pc.setFactorSolverType('mumps')
    opts = PETSc.Options("mat_mumps_")
    opts["icntl_7"] = 4
    # opts["icntl_8"] = 8
    opts["icntl_4"] = 1
    opts["icntl_11"] = 2
    # opts["icntl_18"] = 3
    opts["icntl_10"] = 4
    opts["cntl_3"] = 1e-6

    ###################################
    ## Eigenvalue Solver Options
    ###################################
    # tname = 'arnoldi'
    # E.setType(tname)
    E.setTolerances(tol=1e-8, max_it=15000)
    E.setDimensions(nev, ncv, mpd)
    E.setWhichEigenpairs(E.Which.TARGET_MAGNITUDE)
    E.setTarget(complex(the_shift))
    E.setFromOptions()

    if (rank==0):
        print ''
        print ' TYPE OF KSP: ', K.getType()
        print ' ST Type = ', ST.getType()
        print ' ST Shift = ', ST.getShift()
        print ' Target = ', E.getTarget()
        print ' EGV Solver = ', E.getType()
        print ' Region of Spectrum = ', E.getWhichEigenpairs()
        print ' Number of eigenvalues requested = ', nev
        print ' Number of column vectors = ', ncv
        print ' Maximum dimension of projected problem = ', mpd
        print ''
    E.solve()

    #######################################
    ## POST-PROCESSING
    #######################################
    its = E.getIterationNumber()
    tol, max_it = E.getTolerances()
    ksp_its = K.getIterationNumber()
    nconv = E.getConverged()

    if (rank==0):
        print ''
        print ' Number of iterations of the EPS method: ', its
        print ' CRITERIA: tol= ', tol,', maxit= ',max_it
        print ' Number of iterations of the ksp method: ', ksp_its
        print ' Number of requested eigenvalues: ', nev
        print ' Number of converged eigenvalues: ', nconv

    # Create the results vectors
    xr, tmp = A.getVecs()
    xi, tmp = A.getVecs()
    eigs = []
    if nconv > 0:
        Print("")
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

            eigvecfile = './RESULTS/eigf_'+str(i)+'.pval'
            scatter, eigenvec = PETSc.Scatter.toZero(xr)
            scatter.scatter(xr, eigenvec, False, PETSc.Scatter.Mode.FORWARD)
            if (rank==0):
                if (adjoint):
                    eigenvec = mmatinv.dot(eigenvec)
                mode2pval(eigvecfile, eigenvec, nvars, n, neq, local_id, beta, dreduced, rgid)
                if (beta!=0):
                    mode2pval3D(eigvecfile, eigenvec, nvars, n, neq, beta, 21, dreduced, rgid)


        Print("")

    if (rank==0):
        if(adjoint):
            eigv_file = './RESULTS/eigv_ADJ.dat'
        else:
            eigv_file = './RESULTS/eigv_DIR.dat'

        print ' Saving Eigenvalues in ' + eigv_file
        with open(eigv_file, 'w') as w:
            for i in range(nconv):
                w.write('{0:2d}   {1:12.8f}   {2:12.8f}\n'.format(i,
                                                                eigs[i].real,
                                                                eigs[i].imag))
        print ' DONE'
        print ''


if __name__ == "__main__":
    run_slices()
