#! /usr/bin/env python

from netCDF4 import Dataset
import numpy as np
from scipy.sparse import csr_matrix
import sys,os
import pdb

## ================================================================= ##

def opendualgrid(duafile):
    """Opens a netCDF TAU jacobian matrix, and stores it into a CSR matrix."""
    dua = Dataset(duafile)
    dua.set_auto_mask(False)
    npoints = dua.dimensions['nallpoints'].size
    
    # Reading local_id
    local_id = np.zeros(npoints)
    local_id[:] = dua.variables['local_id'][:]
    dua.close()
    return local_id

# -------------------------------------------------------------------- #

def openjacobian(jacfile):
    """Opens a netCDF TAU jacobian matrix, and stores it into a CSR matrix."""
    jac = Dataset(jacfile)
    nnz = jac.dimensions['nnz'].size
    n   = jac.dimensions['nvars'].size - 1
    neq = jac.dimensions['neq'].size

    # Reading and index correction (from Fortran to C)
    row_ptr = jac.variables['row_ptr'][:] - 1
    col_ind = jac.variables['col_ind'][:] - 1
    data = jac.variables['data'][:]
    jac.close()
    mjac = csr_matrix((data, col_ind, row_ptr), shape=(n,n))
    mjac.eliminate_zeros()
    return mjac, neq

# -------------------------------------------------------------------- #

def openqe(jacfile):
    """Opens a netCDF TAU jacobian matrix, and stores it into a CSR matrix."""
    jac = Dataset(jacfile)
    nnz = jac.dimensions['nnz'].size
    n   = jac.dimensions['nvars'].size - 1
    neq = jac.dimensions['neq'].size

    # Reading and index correction (from Fortran to C)
    row_ptr = jac.variables['row_ptr'][:]
    col_ind = jac.variables['col_ind'][:]
    data = jac.variables['data'][:]
    jac.close()
    mjac = csr_matrix((data, col_ind, row_ptr), shape=(n,n))
    # mjac.eliminate_zeros()
    return mjac, neq

# -------------------------------------------------------------------- #

def openegvec(modefile, neq):
    """Opens a global mode from zTAUev and stores it into a 1D array."""
    mode = Dataset(modefile)
    gridpoints = mode.dimensions['no_of_points'].size/2
    n = gridpoints*neq
    q = np.zeros(n, dtype='c16')
    rho = mode.variables['rho'][:] + 1j*mode.variables['rho_i'][:]
    u   = mode.variables['u'][:]   + 1j*mode.variables['u_i'][:]
    w   = mode.variables['w'][:]   + 1j*mode.variables['w_i'][:]
    e   = mode.variables['e'][:]   + 1j*mode.variables['e_i'][:]
    if neq>4:
        t1  = mode.variables['turb1'][:] + 1j*mode.variables['turb1_i'][:]
        t2  = mode.variables['turb2'][:] + 1j*mode.variables['turb2_i'][:]
    gid = mode.variables['global_id'][:]
    for i in range(0, n, neq):
        q[i]   = rho[i/neq]
        q[i+1] = u[i/neq]
        q[i+2] = w[i/neq]
        q[i+3] = e[i/neq]
        if neq>4:
            q[i+4] = t1[i/neq]
            q[i+5] = t2[i/neq]
    mode.close()
    return q, gid

# -------------------------------------------------------------------- #

def openresidual(resfile, neq):
    """Opens a residuals file from TAU and stores it into a 1D array."""
    mode = Dataset(resfile)
    gridpoints = mode.dimensions['no_of_points'].size/2
    n = gridpoints*neq
    q = np.zeros(n, dtype='f8')
    rho = mode.variables['density_residual'][:]
    u   = mode.variables['x-velocity_residual'][:]
    w   = mode.variables['z-velocity_residual'][:]
    e   = mode.variables['energy_residual'][:]
    if neq>4:
        t1  = mode.variables['k_residual'][:]
        t2  = mode.variables['omega_residual'][:]

    for i in range(0, n, neq):
        q[i]   = rho[i/neq]
        q[i+1] = u[i/neq]
        q[i+2] = w[i/neq]
        q[i+3] = e[i/neq]
        if neq>4:
            q[i+4] = t1[i/neq]
            q[i+5] = t2[i/neq]
    mode.close()
    return q

# -------------------------------------------------------------------- #

def openbflow(resfile, neq):
    """Opens a residuals file from TAU and stores it into a 1D array."""
    mode = Dataset(resfile)
    gridpoints = mode.dimensions['no_of_points'].size/2
    n = gridpoints*neq
    q = np.zeros(n, dtype='f8')
    rho = mode.variables['density'][:]
    u   = mode.variables['x_velocity'][:]
    w   = mode.variables['z_velocity'][:]
    e   = mode.variables['pressure'][:]
    gid = mode.variables['global_id'][:]
    if neq>4:
        t1  = mode.variables['turb_kinetic_energy'][:]
        t2  = mode.variables['turb_omega'][:]

    for i in range(0, n, neq):
        q[i]   = rho[i/neq]
        q[i+1] = u[i/neq]
        q[i+2] = w[i/neq]
        q[i+3] = e[i/neq]
        if neq>4:
            q[i+4] = t1[i/neq]
            q[i+5] = t2[i/neq]
    mode.close()
    return q, gid

# -------------------------------------------------------------------- #

def read_coordinates(coordfile, rlength, beta):
    """Read coordinates from TAU coo file."""
    print(' READING COORDINATES FROM COORD FILE: ', coordfile)
    with open(coordfile) as f:
        ndof, ndim = f.readline().split()
        data = f.readlines()
    data = [line.split() for line in data]
    data = np.array( [map(float,line) for line in data] )
    data *= rlength ## SCALING WITH REYNOLDS LENGTH
    
    # NEQ counter
    neq = 1
    prev = data[0,:]
    for i in range(1,data.shape[0]):
        new = data[i,:]
        if np.array_equal(prev,new):
            neq+=1
        else:
            break
    print(' Number of equations in coordinates file = ', neq)
    
    if (beta==0.):
        return data
    else:
        print(' Correcting number of equations in coordinates file...')
        coord = np.zeros((data.shape[0]/neq, data.shape[1]))
        for pt in range(0, data.shape[0]/neq):
            coord[pt,:] = data[pt*neq,:]
        neq += 1
        new_data = np.zeros((coord.shape[0]*neq, coord.shape[1]))
        for pt in range(coord.shape[0]*neq):
            new_data[pt,:] = coord[pt/neq,:]
            
        return new_data
                
#    return coord

# -------------------------------------------------------------------- #

def opensensitivity(sensfile, neq):
    """Opens a sensitivity file and stores it into a 1D array."""
    mode = Dataset(sensfile)
    gridpoints = mode.dimensions['no_of_points'].size/2
    n = gridpoints*neq
    sens = np.zeros(n, dtype='c16')
    rho = mode.variables['rho'][:] + 1j*mode.variables['rho_i'][:]
    u   = mode.variables['u'][:]   + 1j*mode.variables['u_i'][:]
    w   = mode.variables['w'][:]   + 1j*mode.variables['w_i'][:]
    e   = mode.variables['e'][:]   + 1j*mode.variables['e_i'][:]
    if neq>4:
        t1  = mode.variables['t1'][:] + 1j*mode.variables['t1_i'][:]
        t2  = mode.variables['t2'][:] + 1j*mode.variables['t2_i'][:]

    for i in range(0, n, neq):
        sens[i]   = rho[i/neq]
        sens[i+1] = u[i/neq]
        sens[i+2] = w[i/neq]
        sens[i+3] = e[i/neq]
        if neq>4:
            sens[i+4] = t1[i/neq]
            sens[i+5] = t2[i/neq]
    mode.close()
    return sens

# -------------------------------------------------------------------- #
