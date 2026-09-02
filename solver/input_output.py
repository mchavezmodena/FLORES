#! /usr/bin/env python

from netCDF4 import Dataset
import numpy as np
import h5py
from scipy.sparse import csr_matrix, csc_matrix
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

def open_sod2d_jacobian(jacfile):
    # read from SOD2D's WIP hdf format

    f = h5py.File(jacfile,'r')

    col_ptr = np.array(f['col_ptr'][:],dtype=np.int32)
    row_ind = np.array(f['row_ind'][:],dtype=np.int32)
    values = np.array(f['values'][:],dtype=np.float64)
    coords = np.array(f['node_coords'][:],dtype=np.float64)

    n = len(col_ptr)-1
    numpoints = np.shape(coords)[0]

    neq = n/numpoints # number of eqs = number of degrees of freedom / number of nodes

    mjac = csc_matrix((values,row_ind,col_ptr))
    mjac = mjac.tocsr()
    mjac.eliminate_zeros()

    return mjac, neq

def open_split_sod2d_jacobian(jacfiles):
    # read from SOD2D's WIP hdf format
    # uses the "split parallel" format where multiple hdfs contain a single matrix

    sorted_files = sorted(jacfiles)

    nnz = 0
    num_cols = 0
    for jacfile in jacfiles:
    
        f = h5py.File(jacfile,'r')
        nnz += f['col_ptr'][-1]
        num_cols += len(f['col_ptr'][:])-1 # TODO might fail
        f.close()

    col_ptr = np.zeros(num_cols,dtype=np.int32)
    row_ind = np.zeros(nnz,dtype=np.int32)
    values = np.zeros(nnz,dtype=np.float64)
    coords = np.zeros(num_cols//5,dtype=np.float64) # TODO might be wrong

    previous_col = 0
    current_col = 0
    previous_nnz = 0
    current_nnz = 0
    for jacfile in jacfiles:
    
        f = h5py.File(jacfile,'r')
        current_nnz = previous_nnz + f['col_ptr'][-1]
        current_col = previous_col+len(f['col_ptr'][:])-1
        col_ptr[previous_col:current_col-1] = previous_nnz + f['col_ptr'][:]
        row_ind[previous_nnz:current_nnz-1] = f['row_ind'][:]
        values[previous_nnz:current_nnz-1] = f['values'][:]
        coords[previous_col//5:current_col//5] = f['node_coords'][:]

        f.close()

    n = len(col_ptr)-1
    numpoints = np.shape(coords)[0]

    neq = n/numpoints # number of eqs = number of degrees of freedom / number of nodes

    mjac = csc_matrix((values,row_ind,col_ptr))
    mjac = mjac.tocsr()
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
    """Read coordinates from TAU coo file.

    Optimised version: uses numpy.loadtxt (C-level parser) instead of
    Python-level readlines/map/list-comprehension, and replaces all
    O(n) Python loops with vectorised numpy operations.

    Speedup vs original: typically 10-50x for large meshes.
    """
    print(' READING COORDINATES FROM COORD FILE: ', coordfile)

    # ── Read header (first line) and data in one C-level call ────────────
    with open(coordfile) as f:
        ndof, ndim = f.readline().split()   # header: total_rows  n_dims

    # numpy.loadtxt is implemented in C and is 10-50x faster than
    # Python-level readlines + map(float, ...) for large files.
    data = np.loadtxt(coordfile, dtype=np.float64, skiprows=1)
    data *= rlength   # scale with reference length

    # ── Detect neq: count leading duplicate coordinate rows ──────────────
    # Vectorised: compare all rows against row 0 simultaneously,
    # find first row that differs — no Python loop needed.
    matches = np.all(data == data[0, :], axis=1)   # bool array
    # argmin returns index of first False (first non-matching row)
    first_diff = int(np.argmin(matches))
    neq = first_diff if first_diff > 0 else 1
    print(' Number of equations in coordinates file = ', neq)

    if beta == 0.0:
        return data

    # ── beta != 0: deduplicate and expand to neq+1 equations ─────────────
    print(' Correcting number of equations in coordinates file...')

    # Extract one row per grid point (every neq-th row starting at 0)
    coord = data[::neq, :]                          # shape (gridpoints, ndim)

    # Expand: each grid point repeated (neq+1) times — pure numpy, no loop
    new_data = np.repeat(coord, neq + 1, axis=0)   # shape (gridpoints*(neq+1), ndim)

    return new_data

def read_sod2d_coordinates(coordfile, rlength, beta):
    """Read coordinates from sod2d hdf5 file.

    Uses h5py to read the coordinates from SOD2D.
    """
    print(' READING COORDINATES FROM SOD2D OUTPUT FILE: ', coordfile)

    f = h5py.File(coordfile,'r')
    coords = np.array(f['node_coords'][:],dtype=np.float64)
    coords *= rlength

    neq = 5

    new_data = np.repeat(coords, neq, axis=0)   # shape (gridpoints*(neq+1), ndim)

    return new_data

def dump_sod2d_coordinates(filename: str, coords: np.ndarray, neq: int, kept_idx = None):
    """
    Dump the coords read from SOD2D to an output file matching the
    .coo format from TAU

    This makes visualisation easier later
    """
    print(' DUMPING SOD2D COORDS TO FILE: ', filename)
    if(type(kept_idx) is np.ndarray):
        new_coords = coords[kept_idx,:]
    else:
        new_coords = coords
    with open(filename,"w") as f:
        coords_shape = np.shape(new_coords)
        f.write(f"{coords_shape[0]} {coords_shape[1]} \n")
        for i in range(0, coords_shape[0]):
            f.write(f"{new_coords[i,0]} {new_coords[i,1]} {new_coords[i,2]} \n")


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