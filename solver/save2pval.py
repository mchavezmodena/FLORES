#! /usr/bin/env python

import sys, os
from netCDF4 import Dataset
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Internal helper
# ─────────────────────────────────────────────────────────────────────────────

def _write_var(nc, name, first_half, second_half=None):
    """
    Create a float64 variable on 'no_of_points' and fill both halves.

    Parameters
    ----------
    nc          : open Dataset
    name        : variable name
    first_half  : array of length gridpoints (first half of the TAU layout)
    second_half : array of length gridpoints, or None to mirror first_half
    """
    nc.createVariable(name, 'f8', ('no_of_points',))
    gp = len(first_half)
    nc.variables[name][:gp] = first_half
    nc.variables[name][gp:] = first_half if second_half is None else second_half


def _unpack_sol(sol, neq, is_simulator_sod2d):
    """
    Unpack a flat complex solution vector into per-variable arrays via
    strided slicing (no Python loops).

    Returns a dict with keys: rho, u, w, e, [turb1], [turb2]
    """
    if(is_simulator_sod2d):
        d = {
        'rho':  sol[0::neq],
        'u':    sol[1::neq],
        'v':    sol[2::neq],
        'w':    sol[3::neq],
        'e':    sol[4::neq],
        }
        return d   
    d = {
        'rho':  sol[0::neq],
        'u':    sol[1::neq],
        'w':    sol[2::neq],
        'e':    sol[3::neq],
    }
    if neq >= 5:
        d['turb1'] = sol[4::neq]
    if neq >= 6:
        d['turb2'] = sol[5::neq]
    return d


############################################################################
# sens2pval
############################################################################

def sens2pval(filename, gid, sens, npoints, neq, dreduced=False, rgid=None):
    """
    Write a sensitivity field to a TAU-compatible netCDF .pval file.

    Note: the 'v' (spanwise) component is zero for 2-D cases (beta=0).
    """
    senspoints = npoints // neq
    nprob      = len(gid)
    gridpoints = nprob // 2

    # Unpack — vectorised strided slicing, no loops
    rho = sens[0::neq].copy()
    u   = sens[1::neq].copy()
    w   = sens[2::neq].copy()
    e   = sens[3::neq].copy()
    v   = np.zeros(senspoints, dtype='c16')   # spanwise: zero for 2-D

    if neq >= 5:
        t1 = sens[4::neq].copy()
    if neq >= 6:
        t2 = sens[5::neq].copy()

    sens_R  = np.sqrt(u.real**2 + w.real**2)
    sens_Im = np.sqrt(u.imag**2 + w.imag**2)

    amg_f = Dataset(filename, 'w')
    amg_f.createDimension('no_of_points', nprob)

    _write_var(amg_f, 'global_id_f8',  # global_id kept as int below
               np.zeros(gridpoints))   # placeholder — overwritten
    amg_f.createVariable('global_id', 'i', ('no_of_points',))
    amg_f.variables['global_id'][:] = gid

    if dreduced:
        # Scatter reduced values into full arrays, then mirror
        def _scatter(arr_r, arr_i=None):
            full_r = np.zeros(gridpoints)
            full_r[rgid] = arr_r
            if arr_i is not None:
                full_i = np.zeros(gridpoints)
                full_i[rgid] = arr_i
                return full_r, full_i
            return full_r

        _write_var(amg_f, 'rho',    *_scatter(rho.real, rho.imag))
        _write_var(amg_f, 'rho_i',  np.zeros(gridpoints), _scatter(rho.imag))  # mirror
        # rebuild cleanly
        for vname, arr in [('rho', rho), ('u', u), ('w', w), ('e', e),
                           ('v', v)]:
            fr = np.zeros(gridpoints); fr[rgid] = arr.real
            fi = np.zeros(gridpoints); fi[rgid] = arr.imag
            _write_var(amg_f, vname,       fr)
            if vname != 'v':
                _write_var(amg_f, vname+'_i', fi)

        sr_full = np.zeros(gridpoints); sr_full[rgid] = sens_R
        si_full = np.zeros(gridpoints); si_full[rgid] = sens_Im
        _write_var(amg_f, 'sens_R',  sr_full)
        _write_var(amg_f, 'sens_Im', si_full)

        if neq >= 5:
            ft1r = np.zeros(gridpoints); ft1r[rgid] = t1.real
            ft1i = np.zeros(gridpoints); ft1i[rgid] = t1.imag
            _write_var(amg_f, 'turb1',   ft1r)
            _write_var(amg_f, 'turb1_i', ft1i)
        if neq >= 6:
            ft2r = np.zeros(gridpoints); ft2r[rgid] = t2.real
            ft2i = np.zeros(gridpoints); ft2i[rgid] = t2.imag
            _write_var(amg_f, 'turb2',   ft2r)
            _write_var(amg_f, 'turb2_i', ft2i)
    else:
        for vname, arr in [('rho', rho), ('u', u), ('w', w), ('e', e),
                           ('v', v)]:
            _write_var(amg_f, vname, arr.real)
            if vname != 'v':
                _write_var(amg_f, vname+'_i', arr.imag)

        _write_var(amg_f, 'sens_R',  sens_R)
        _write_var(amg_f, 'sens_Im', sens_Im)

        if neq >= 5:
            _write_var(amg_f, 'turb1',   t1.real)
            _write_var(amg_f, 'turb1_i', t1.imag)
        if neq >= 6:
            _write_var(amg_f, 'turb2',   t2.real)
            _write_var(amg_f, 'turb2_i', t2.imag)

    amg_f.close()
    return np.array(sens_R + 1j * sens_Im)


############################################################################
# sol2pval
############################################################################

def sol2pval(filename, gid, sol, npoints, neq, dreduced=False, rgid=None):
    """Write a real base-flow solution to a TAU .pval file."""
    gridpoints = npoints // neq

    # Vectorised unpacking — replaces the Python xrange loop
    rho = np.asarray(sol[0::neq], dtype=np.float64)
    u   = np.asarray(sol[1::neq], dtype=np.float64)
    w   = np.asarray(sol[2::neq], dtype=np.float64)
    e   = np.asarray(sol[3::neq], dtype=np.float64)
    v   = np.zeros(gridpoints,    dtype=np.float64)   # spanwise: zero for 2-D

    amg_f = Dataset(filename, 'w', format='NETCDF3_64BIT_OFFSET')
    amg_f.createDimension('no_of_points', gridpoints * 2)

    amg_f.createVariable('global_id', 'i', ('no_of_points',))
    amg_f.variables['global_id'][:] = gid

    fields = [('density', rho), ('x_velocity', u),
              ('y_velocity', v), ('z_velocity', w), ('pressure', e)]

    if dreduced:
        for vname, arr in fields:
            full = np.zeros(gridpoints)
            full[rgid] = arr
            _write_var(amg_f, vname, full)
    else:
        for vname, arr in fields:
            _write_var(amg_f, vname, arr)

    amg_f.close()


############################################################################
# mode2pval
############################################################################

def mode2pval(filename, sol, npoints, nred, neq, beta=0.0,
              dreduced=False, rgid=None, is_simulator_sod2d=False):
    """
    Write a complex global mode to a TAU-compatible netCDF .pval file.

    Parameters
    ----------
    filename : output path
    sol      : PETSc Vec or numpy array (complex, length nred)
    npoints  : total grid points * neq  (full mesh, for output sizing)
    nred     : number of DOFs in the (possibly reduced) solution vector
    neq      : number of equations per grid point
    beta     : spanwise wavenumber (informational only here)
    dreduced : True if domain reduction was applied
    rgid     : reduced grid point indices (used when dreduced=True)
    """
    # Convert PETSc Vec to numpy if needed
    if hasattr(sol, 'getArray'):
        sol = sol.getArray()
    sol = np.asarray(sol, dtype=np.complex128)

    vars_red = _unpack_sol(sol, neq, is_simulator_sod2d)   # dict: rho, u, w, e, [turb1, turb2]

    gridpoints_out = npoints // neq    # full mesh size for output file

    amg_f = Dataset(filename, 'w', format='NETCDF3_64BIT_OFFSET')
    amg_f.createDimension('no_of_points', gridpoints_out * 2)

    amg_f.createVariable('global_id', 'i', ('no_of_points',))
    amg_f.variables['global_id'][:] = np.arange(gridpoints_out * 2, dtype='i4')

    
    base_vars = ['rho', 'u', 'w', 'e']
    turb_vars = []
    if neq >= 5:
        turb_vars.append('turb1')
    if neq >= 6:
        turb_vars.append('turb2')
    all_vars = base_vars + turb_vars
    if(is_simulator_sod2d):
        base_vars = ['rho', 'u', 'v', 'w', 'e']
        turb_vars = []
        all_vars = base_vars + turb_vars
    if dreduced:
        # Scatter reduced-domain values into full-size arrays
        for vname in all_vars:
            full = np.zeros(gridpoints_out, dtype=np.complex128)
            full[rgid] = vars_red[vname]
            _write_var(amg_f, vname,       full.real)
            _write_var(amg_f, vname + '_i', full.imag)
    else:
        gp_red = nred // neq
        for vname in all_vars:
            arr = vars_red[vname]
            # Pad with zeros if reduced size < full size (shouldn't happen
            # in the non-dreduced path, but guard against shape mismatch)
            if len(arr) < gridpoints_out:
                full = np.zeros(gridpoints_out, dtype=np.complex128)
                full[:len(arr)] = arr
            else:
                full = arr
            _write_var(amg_f, vname,       full.real)
            _write_var(amg_f, vname + '_i', full.imag)

    amg_f.close()


############################################################################
# mode2pval3D
############################################################################

def mode2pval3D(filename, sol, npoints, nred, neq, beta, nums,
                dreduced=False, rgid=None):
    """
    Write a 3-D reconstructed mode (beta != 0) to a TAU .pval file.

    The spanwise expansion uses numpy broadcasting instead of nested
    Python loops.
    """
    if hasattr(sol, 'getArray'):
        sol = sol.getArray()
    sol = np.asarray(sol, dtype=np.complex128)

    gridpoints_red = nred  // neq
    gridpoints_out = npoints // neq

    # Unpack — vectorised
    rho = sol[0::neq]
    u   = sol[1::neq]
    w   = sol[2::neq]
    e   = sol[3::neq]
    if beta != 0.0:
        v = sol[3::neq]   # for beta!=0 layout: rho,u,v,w,e
        w = sol[4::neq] if neq >= 5 else np.zeros(gridpoints_red, dtype=np.complex128)
        e = sol[4::neq] if neq >= 5 else sol[3::neq]
        # Correct unpacking for beta!=0: rho,u,v,w,e
        rho = sol[0::neq]
        u   = sol[1::neq]
        v   = sol[2::neq]
        w   = sol[3::neq]
        e   = sol[4::neq] if neq >= 5 else np.zeros(gridpoints_red, dtype=np.complex128)
    else:
        v = np.zeros(gridpoints_red, dtype=np.complex128)

    # Scatter to full mesh if domain-reduced
    def _to_full(arr):
        if dreduced:
            full = np.zeros(gridpoints_out, dtype=np.complex128)
            full[rgid] = arr
            return full
        if len(arr) < gridpoints_out:
            full = np.zeros(gridpoints_out, dtype=np.complex128)
            full[:len(arr)] = arr
            return full
        return arr

    rho = _to_full(rho)
    u   = _to_full(u)
    v   = _to_full(v)
    w   = _to_full(w)
    e   = _to_full(e)

    out_file = filename[:-5] + '3D.pval' if filename.endswith('.pval') else filename[:-4] + '3D.pval'
    amg_f = Dataset(out_file, 'w', format='NETCDF3_64BIT_OFFSET')
    amg_f.createDimension('no_of_points', gridpoints_out * nums)

    amg_f.createVariable('global_id', 'i', ('no_of_points',))
    amg_f.variables['global_id'][:] = np.arange(gridpoints_out * nums, dtype='i4')

    # Spanwise phase expansion — vectorised with numpy broadcasting
    # slice_phases shape: (nums,) ; values: exp(i * beta * y_slice)
    Ly           = 2.0 * np.pi / beta if beta != 0.0 else 1.0
    y_slices     = np.arange(nums) * (Ly / nums)           # (nums,)
    phases       = np.exp(1j * beta * y_slices)            # (nums,)

    for ncname, base_arr in [('rho', rho), ('u', u), ('v', v),
                              ('w', w),    ('e', e)]:
        amg_f.createVariable(ncname, 'f8', ('no_of_points',))
        # base_arr shape: (gridpoints_out,)
        # expanded:       (nums, gridpoints_out) via outer product, then flatten
        expanded = np.real(
            np.outer(phases, base_arr)    # (nums, gridpoints_out)
        ).ravel()                         # (nums * gridpoints_out,)
        amg_f.variables[ncname][:] = expanded

    amg_f.close()