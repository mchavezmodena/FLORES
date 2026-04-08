#! /usr/bin/env python
# This file only runs in sequential mode

import numpy as np
# import matplotlib.pyplot as plt
import sys,os
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

def run_slices():
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    Print = PETSc.Sys.Print

    #######################
    jacfile = '/home/ivan/Nextcloud/TRANSDIFFUSE/BFS_MChavez/jac_Ivan/samg.matrix.amg.pval'
    volfile = '/home/ivan/Nextcloud/TRANSDIFFUSE/BFS_MChavez/jac_Ivan/samg.matrix.vol'    # Not used unless gen = True
    #######################
    beta = 0.0
    mach = 0.1  # 0.2 for cyl case
    rlength = 1
    #######################
    dreduced = False
    # xmin = -10
    # xmax = 10
    # zmin = -5
    # zmax = 5
    xmin = 0.12
    xmax = 0.35
    zmin = -0.05
    zmax = 0.05
    #######################
    nev = 10
    gen = False     # If the Jacobian is already scaled by the cell volumes, then solve a standard EVP, not a generalized EVP (no mass matrix needed)
    adjoint = False
    #the_shift = -0.05 + 0.115*2.0*np.pi*1j     # For cyl case
    the_shift = -0.05 + 1.0*1j
    #######################
    
    fac = 1. / (mach*np.sqrt(1.4))  # Eq. (3.19) Alex PhD thesis

    if ( os.path.isdir('./eig_results') ):
        pass
    else:
        os.mkdir('./eig_results')

    Print('')
    Print(' Reading Jacobian')
    amatrix, neq = openjacobian(jacfile)
    amatrix.data *= fac
    nvars = amatrix.shape[0]
    Print(' Matrix main dimension = {0}'.format(nvars) )
    Print(' Number of equations = {0}'.format(neq))
    Print('')

    gridpoints = int(nvars/neq)

    ## READING MASS MATRIX AND STORE IT IN SPARSE FORMAT ##
    Print (' Reading mass matrix and generating M')
    Print ('')
    one = 1. + 0j
    with open(volfile, 'r') as f:
        vols = f.readlines()
        vols = [float(line) for line in vols]
    bmatrix    = identity(nvars, dtype='c16', format='csr')  # Matrix M
    bmatrix.data[:] = [vols[i//neq]*one for i in range(nvars)]
    
    ###### DOMAIN REDUCTION #####
    if (dreduced==True):
        Print('')
        Print(' Applying domain reduction')
        Print(' XMIN/XMAX = {0}/{1}'.format(xmin,xmax))
        Print(' ZMIN/ZMAX = {0}/{1}'.format(zmin,zmax))
        # 1. Read coordinates from file
        coord = read_coordinates('/home/ivan/Nextcloud/TRANSDIFFUSE/TAU_cyl/run_with_Alexdev/jac/samg.matrix.coo', rlength, beta)
        # 2. Generate the permutation matrix
        dr = domain_reduction(zmin, zmax, xmin, xmax)
        dr.create_Pmatrix(coord)
        # 3. Reorder and slice the jacobian and volumes matrix
        Print(' Previous number of NNZ elem = {0}'.format(amatrix.nnz))
        amatrix = dr.reduce_matrix(amatrix)
        Print(' New number of NNZ elem = {0}'.format(amatrix.nnz))
        bmatrix = dr.reduce_matrix(bmatrix)
        # mmatinv = dr.reduce_matrix(mmatinv)
        n = amatrix.shape[0]
        Print(' New leading dimension of A Matrix = {0}'.format(n))
        # 5. Generate a local_id vector, reorder and slice it
        #    (this will be needed for output indexing)
        localid = np.arange(0, gridpoints, 1, dtype='i4')
        localid = np.repeat(localid, neq)
        rgid = dr.reduce_vector(localid)
        rgid = rgid[0::neq].astype(int)
        Print('')
    else:
        rgid = None
        n = nvars

 
    ## CREATING PETSc MATRICES
    A = PETSc.Mat()
    A.create(PETSc.COMM_WORLD)
    A.setSizes([n, n])
    A.setUp()
    A.setType('seqaij')
   
    Print (' Assembling PETSc matrix...')
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
        B[pt,pt] = bmatrix.data[pt]
    B.assemble()

    ######################################################
    ######################################################
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
        # E.setProblemType( SLEPc.EPS.ProblemType.NHEP )
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
    opts["icntl_6"] = 2
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
    # E.setTwoSided()
    E.setFromOptions()

    if rank == 0:
        print('')
        print(' TYPE OF KSP: ', K.getType())
        print(' ST Type = ', ST.getType())
        print(' ST Shift = ', ST.getShift())
        print(' Target = ', E.getTarget())
        print(' EGV Solver = ', E.getType())
        print(' EPS Problem Type = ', E.getProblemType())
        print(' Region of Spectrum = ', E.getWhichEigenpairs())
        print(' Number of eigenvalues requested = ', nev)
        print(' Number of column vectors = ', ncv)
        print(' Maximum dimension of projected problem = ', mpd)
        print('')
    E.solve()

    #######################################
    ## POST-PROCESSING
    #######################################
    its = E.getIterationNumber()
    tol, max_it = E.getTolerances()
    ksp_its = K.getIterationNumber()
    nconv = E.getConverged()

    if (rank==0):
        print('')
        print(' Number of iterations of the EPS method: ', its)
        print(' CRITERIA: tol= ', tol,', maxit= ',max_it)
        print(' Number of iterations of the ksp method: ', ksp_its)
        print(' Number of requested eigenvalues: ', nev)
        print(' Number of converged eigenvalues: ', nconv)

    # Create the results vectors
    xr, xrl = A.getVecs()
    xi, xil = A.getVecs()
    eigs = []
    if nconv > 0:
        Print("")
        Print("           k           ||Ax-kx||/||kx|| ")
        Print("--------------------- ------------------")
        for i in range(nconv):
            k = E.getEigenpair(i, xr, xi)
            # E.getLeftEigenvector(i, xrl, xil)
            eigs.append(k)
            error = E.computeError(i)
            if k.imag != 0.0:
              Print(" %9f%+9f j    %12g" % (k.real, k.imag, error))
            else:
              Print(" %12f         %12g" % (k.real, error))

            eigvecfile = './eig_results/eigf_'+str(i)+'.pval'
            scatter, eigenvec = PETSc.Scatter.toZero(xr)
            scatter.scatter(xr, eigenvec, False, PETSc.Scatter.Mode.FORWARD)
            if (rank==0):
                mode2pval(eigvecfile, eigenvec, nvars, n, neq, beta, dreduced, rgid)
                if (beta!=0):
                    mode2pval3D(eigvecfile, eigenvec, nvars, n, neq, beta, 21, dreduced, rgid)


        Print("")

    if (rank==0):
        if(adjoint):
            eigv_file = './eig_results/eigv_ADJ.dat'
        else:
            eigv_file = './eig_results/eigv_DIR.dat'

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
