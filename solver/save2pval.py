#! /usr/bin/env python

import sys,os
from netCDF4 import Dataset
import numpy as np

def sens2pval(filename, gid, sens, npoints, neq, dreduced=False, rgid=None):
    senspoints = npoints//neq
    nprob = len(gid)
    gridpoints = nprob//2

    rho = np.zeros(senspoints, dtype='c16')
    u   = np.zeros(senspoints, dtype='c16')
    w   = np.zeros(senspoints, dtype='c16')
    e   = np.zeros(senspoints, dtype='c16')
    # if neq>4:
    #     t1 = np.zeros(senspoints, dtype='c16')
    #     t2 = np.zeros(senspoints, dtype='c16')

    rho[:] = sens[0::neq]
    u[:]   = sens[1::neq]
    w[:]   = sens[2::neq]
    e[:]   = sens[3::neq]
        # if neq>4:
        #     t1[i]  = sens[i*neq+4]
        #     t2[i]  = sens[i*neq+5]

    sens_R  = np.sqrt(u.real**2 + w.real**2)
    sens_Im = np.sqrt(u.imag**2 + w.imag**2)

    # rho[:] /= np.max(rho[:])
    # u[:] /= np.max(u[:])
    # w[:] /= np.max(w[:])
    # e[:] /= np.max(e[:])
    # sens_R[:] /= np.max(sens_R)
    # sens_Im[:] /= np.max(sens_Im)

    ## NETCDF SENSITIVITY OUTPUT ##
    amg_f = Dataset(filename, 'w')
    amg_f.createDimension('no_of_points', nprob)

    amg_f.createVariable('global_id', 'i', ('no_of_points',))
    amg_f.variables['global_id'][:] = gid

    if (dreduced==True):
        amg_f.createVariable('rho', 'f8', ('no_of_points',))
        amg_f.variables['rho'][:] = np.zeros(gridpoints*2)
        amg_f.variables['rho'][rgid] = np.real(rho[:])
        amg_f.variables['rho'][gridpoints:] = amg_f.variables['rho'][:gridpoints]
        amg_f.createVariable('rho_i', 'f8', ('no_of_points',))
        amg_f.variables['rho_i'][:] = np.zeros(gridpoints*2)
        amg_f.variables['rho_i'][rgid] = np.imag(rho[:])
        amg_f.variables['rho_i'][gridpoints:] = amg_f.variables['rho_i'][:gridpoints]

        amg_f.createVariable('u', 'f8', ('no_of_points',))
        amg_f.variables['u'][:] = np.zeros(gridpoints*2)
        amg_f.variables['u'][rgid] = np.real(u[:])
        amg_f.variables['u'][gridpoints:] = amg_f.variables['u'][:gridpoints]
        amg_f.createVariable('u_i', 'f8', ('no_of_points',))
        amg_f.variables['u_i'][:] = np.zeros(gridpoints*2)
        amg_f.variables['u_i'][rgid] = np.imag(u[:])
        amg_f.variables['u_i'][gridpoints:] = amg_f.variables['u_i'][:gridpoints]

        amg_f.createVariable('v', 'f8', ('no_of_points',))
        amg_f.variables['v'][:gridpoints] = np.real(v[:])
        amg_f.variables['v'][gridpoints:] = amg_f.variables['v'][:gridpoints]

        amg_f.createVariable('w', 'f8', ('no_of_points',))
        amg_f.variables['w'][:] = np.zeros(gridpoints*2)
        amg_f.variables['w'][rgid] = np.real(w[:])
        amg_f.variables['w'][gridpoints:] = amg_f.variables['w'][:gridpoints]
        amg_f.createVariable('w_i', 'f8', ('no_of_points',))
        amg_f.variables['w_i'][:] = np.zeros(gridpoints*2)
        amg_f.variables['w_i'][rgid] = np.imag(w[:])
        amg_f.variables['w_i'][gridpoints:] = amg_f.variables['w_i'][:gridpoints]

        amg_f.createVariable('e', 'f8', ('no_of_points',))
        amg_f.variables['e'][:] = np.zeros(gridpoints*2)
        amg_f.variables['e'][rgid] = np.real(e[:])
        amg_f.variables['e'][gridpoints:] = amg_f.variables['e'][:gridpoints]
        amg_f.createVariable('e_i', 'f8', ('no_of_points',))
        amg_f.variables['e_i'][:] = np.zeros(gridpoints*2)
        amg_f.variables['e_i'][rgid] = np.imag(e[:])
        amg_f.variables['e_i'][gridpoints:] = amg_f.variables['e_i'][:gridpoints]

        amg_f.createVariable('sens_R', 'f8', ('no_of_points',))
        amg_f.variables['sens_R'][:] = np.zeros(gridpoints*2)
        amg_f.variables['sens_R'][rgid] = sens_R[:]
        amg_f.variables['sens_R'][gridpoints:] = amg_f.variables['sens_R'][:gridpoints]
        amg_f.createVariable('sens_Im', 'f8', ('no_of_points',))
        amg_f.variables['sens_Im'][:] = np.zeros(gridpoints*2)
        amg_f.variables['sens_Im'][rgid] = sens_Im[:]
        amg_f.variables['sens_Im'][gridpoints:] = amg_f.variables['sens_Im'][:gridpoints]

        if neq>4:
            amg_f.createVariable('t1', 'f8', ('no_of_points',))
            amg_f.variables['t1'][:] = np.zeros(gridpoints*2)
            amg_f.variables['t1'][rgid] = np.real(t1[:])
            amg_f.variables['t1'][gridpoints:] = amg_f.variables['t1'][:gridpoints]
            amg_f.createVariable('t1_i', 'f8', ('no_of_points',))
            amg_f.variables['t1_i'][:] = np.zeros(gridpoints*2)
            amg_f.variables['t1_i'][rgid] = np.imag(t1[:])
            amg_f.variables['t1_i'][gridpoints:] = amg_f.variables['t1_i'][:gridpoints]

            amg_f.createVariable('t2', 'f8', ('no_of_points',))
            amg_f.variables['t2'][:] = np.zeros(gridpoints*2)
            amg_f.variables['t2'][rgid] = np.real(t2[:])
            amg_f.variables['t2'][gridpoints:] = amg_f.variables['t2'][:gridpoints]
            amg_f.createVariable('t2_i', 'f8', ('no_of_points',))
            amg_f.variables['t2_i'][:] = np.zeros(gridpoints*2)
            amg_f.variables['t2_i'][rgid] = np.imag(t2[:])
            amg_f.variables['t2_i'][gridpoints:] = amg_f.variables['t2_i'][:gridpoints]
    else:
        amg_f.createVariable('rho', 'f8', ('no_of_points',))
        amg_f.variables['rho'][:gridpoints] = np.real(rho[:])
        amg_f.variables['rho'][gridpoints:] = amg_f.variables['rho'][:gridpoints]
        amg_f.createVariable('rho_i', 'f8', ('no_of_points',))
        amg_f.variables['rho_i'][:gridpoints] = np.imag(rho[:])
        amg_f.variables['rho_i'][gridpoints:] = amg_f.variables['rho_i'][:gridpoints]

        amg_f.createVariable('u', 'f8', ('no_of_points',))
        amg_f.variables['u'][:gridpoints] = np.real(u[:])
        amg_f.variables['u'][gridpoints:] = amg_f.variables['u'][:gridpoints]
        amg_f.createVariable('u_i', 'f8', ('no_of_points',))
        amg_f.variables['u_i'][:gridpoints] = np.imag(u[:])
        amg_f.variables['u_i'][gridpoints:] = amg_f.variables['u_i'][:gridpoints]

        amg_f.createVariable('v', 'f8', ('no_of_points',))
        amg_f.variables['v'][:gridpoints] = np.real(v[:])
        amg_f.variables['v'][gridpoints:] = amg_f.variables['v'][:gridpoints]

        amg_f.createVariable('w', 'f8', ('no_of_points',))
        amg_f.variables['w'][:gridpoints] = np.real(w[:])
        amg_f.variables['w'][gridpoints:] = amg_f.variables['w'][:gridpoints]
        amg_f.createVariable('w_i', 'f8', ('no_of_points',))
        amg_f.variables['w_i'][:gridpoints] = np.imag(w[:])
        amg_f.variables['w_i'][gridpoints:] = amg_f.variables['w_i'][:gridpoints]

        amg_f.createVariable('e', 'f8', ('no_of_points',))
        amg_f.variables['e'][:gridpoints] = np.real(e[:])
        amg_f.variables['e'][gridpoints:] = amg_f.variables['e'][:gridpoints]
        amg_f.createVariable('e_i', 'f8', ('no_of_points',))
        amg_f.variables['e_i'][:gridpoints] = np.imag(e[:])
        amg_f.variables['e_i'][gridpoints:] = amg_f.variables['e_i'][:gridpoints]

        amg_f.createVariable('sens_R', 'f8', ('no_of_points',))
        amg_f.variables['sens_R'][:gridpoints] = sens_R[:]
        amg_f.variables['sens_R'][gridpoints:] = amg_f.variables['sens_R'][:gridpoints]
        amg_f.createVariable('sens_Im', 'f8', ('no_of_points',))
        amg_f.variables['sens_Im'][:gridpoints] = sens_Im[:]
        amg_f.variables['sens_Im'][gridpoints:] = amg_f.variables['sens_Im'][:gridpoints]

        if neq>4:
            amg_f.createVariable('t1', 'f8', ('no_of_points',))
            amg_f.variables['t1'][:gridpoints] = np.real(t1[:])
            amg_f.variables['t1'][gridpoints:] = amg_f.variables['t1'][:gridpoints]
            amg_f.createVariable('t1_i', 'f8', ('no_of_points',))
            amg_f.variables['t1_i'][:gridpoints] = np.imag(t1[:])
            amg_f.variables['t1_i'][gridpoints:] = amg_f.variables['t1_i'][:gridpoints]

            amg_f.createVariable('t2', 'f8', ('no_of_points',))
            amg_f.variables['t2'][:gridpoints] = np.real(w[:])
            amg_f.variables['t2'][gridpoints:] = amg_f.variables['t2'][:gridpoints]
            amg_f.createVariable('t2_i', 'f8', ('no_of_points',))
            amg_f.variables['t2_i'][:gridpoints] = np.imag(w[:])
            amg_f.variables['t2_i'][gridpoints:] = amg_f.variables['t2_i'][:gridpoints]

    amg_f.close()

    return np.array(sens_R + 1j*sens_Im)

############################################################################

def sol2pval(filename, gid, sol, npoints, neq, dreduced=False, rgid=None):
    gridpoints = npoints/neq
    rho = np.zeros(gridpoints, dtype='f8')
    u   = np.zeros(gridpoints, dtype='f8')
    v   = np.zeros(gridpoints, dtype='f8')
    w   = np.zeros(gridpoints, dtype='f8')
    e   = np.zeros(gridpoints, dtype='f8')
    for i in xrange(0,gridpoints):
        rho[i] = sol[i*4]
        u[i]   = sol[i*4+1]
        w[i]   = sol[i*4+2]
        e[i]   = sol[i*4+3]

    ## NETCDF SENSITIVITY OUTPUT ##
    amg_f = Dataset(filename, 'w')
    amg_f.createDimension('no_of_points', gridpoints*2)

    amg_f.createVariable('global_id', 'i', ('no_of_points',))
    amg_f.variables['global_id'][:] = gid

    if (dreduced==True):
        amg_f.createVariable('density', 'f8', ('no_of_points',))
        amg_f.variables['density'][:] = np.zeros(gridpoints*2)
        amg_f.variables['density'][rgid] = np.real(rho[:])
        amg_f.variables['density'][gridpoints:] = amg_f.variables['density'][:gridpoints]

        amg_f.createVariable('x_velocity', 'f8', ('no_of_points',))
        amg_f.variables['x_velocity'][:] = np.zeros(gridpoints*2)
        amg_f.variables['x_velocity'][rgid] = np.real(u[:])
        amg_f.variables['x_velocity'][gridpoints:] = amg_f.variables['x_velocity'][:gridpoints]

        amg_f.createVariable('y_velocity', 'f8', ('no_of_points',))
        amg_f.variables['y_velocity'][gridpoints] = np.real(v[:])
        amg_f.variables['y_velocity'][gridpoints:] = amg_f.variables['y_velocity'][:gridpoints]

        amg_f.createVariable('z_velocity', 'f8', ('no_of_points',))
        amg_f.variables['z_velocity'][:] = np.zeros(gridpoints*2)
        amg_f.variables['z_velocity'][rgid] = np.real(w[:])
        amg_f.variables['z_velocity'][gridpoints:] = amg_f.variables['z_velocity'][:gridpoints]

        amg_f.createVariable('pressure', 'f8', ('no_of_points',))
        amg_f.variables['pressure'][:] = np.zeros(gridpoints*2)
        amg_f.variables['pressure'][rgid] = np.real(e[:])
        amg_f.variables['pressure'][gridpoints:] = amg_f.variables['pressure'][:gridpoints]

    else:
        amg_f.createVariable('density', 'f8', ('no_of_points',))
        amg_f.variables['density'][:gridpoints] = np.real(rho[:])
        amg_f.variables['density'][gridpoints:] = amg_f.variables['density'][:gridpoints]

        amg_f.createVariable('x_velocity', 'f8', ('no_of_points',))
        amg_f.variables['x_velocity'][:gridpoints] = np.real(u[:])
        amg_f.variables['x_velocity'][gridpoints:] = amg_f.variables['x_velocity'][:gridpoints]

        amg_f.createVariable('y_velocity', 'f8', ('no_of_points',))
        amg_f.variables['y_velocity'][:gridpoints] = np.real(v[:])
        amg_f.variables['y_velocity'][gridpoints:] = amg_f.variables['y_velocity'][:gridpoints]

        amg_f.createVariable('z_velocity', 'f8', ('no_of_points',))
        amg_f.variables['z_velocity'][:gridpoints] = np.real(w[:])
        amg_f.variables['z_velocity'][gridpoints:] = amg_f.variables['z_velocity'][:gridpoints]

        amg_f.createVariable('pressure', 'f8', ('no_of_points',))
        amg_f.variables['pressure'][:gridpoints] = np.real(e[:])
        amg_f.variables['pressure'][gridpoints:] = amg_f.variables['pressure'][:gridpoints]

    amg_f.close()

    return

############################################################################

def mode2pval(filename, sol, npoints, nred, neq, gid, beta=0.0, dreduced=False, rgid=None):
    gridpoints = nred//neq
    rho = np.zeros(gridpoints, dtype='c16')
    u   = np.zeros(gridpoints, dtype='c16')
    w   = np.zeros(gridpoints, dtype='c16')
    e   = np.zeros(gridpoints, dtype='c16')
    
    # if (neq==5):
    #     nu = np.zeros(gridpoints, dtype='c16')
    
    # if (beta==0):
    rho[:] = sol[0::neq]
    u[:]   = sol[1::neq]
    w[:]   = sol[2::neq]
    e[:]   = sol[3::neq]

    if (neq==5):
        turb1 = np.zeros(gridpoints, dtype='c16')
        turb1[:] = sol[4::neq]
    elif (neq==6):
        turb1 = np.zeros(gridpoints, dtype='c16')
        turb2 = np.zeros(gridpoints, dtype='c16')
        turb1[:] = sol[4::neq]
        turb2[:] = sol[5::neq]


    ## NETCDF SENSITIVITY OUTPUT ##
    amg_f = Dataset(filename, 'w', format="NETCDF3_64BIT_OFFSET")
    gridpoints = npoints//neq  ## SOLUTION FILE HAS ALL THE POINTS!
    amg_f.createDimension('no_of_points', gridpoints*2)

    rho_out = np.zeros(gridpoints, dtype='c16')
    u_out   = np.zeros(gridpoints, dtype='c16')
    w_out   = np.zeros(gridpoints, dtype='c16')
    e_out   = np.zeros(gridpoints, dtype='c16')

    if (neq==5):
        turb1_out = np.zeros(gridpoints, dtype='c16')
    elif (neq==6):
        turb1_out = np.zeros(gridpoints, dtype='c16')
        turb2_out = np.zeros(gridpoints, dtype='c16')

    # if (neq==5):
    #     nu_out = np.zeros(gridpoints, dtype='c16')
    
    amg_f.createVariable('global_id', 'i', ('no_of_points',))
    amg_f.variables['global_id'][:] = np.arange(0, gridpoints*2, 1, dtype='i4')
    # amg_f.variables['global_id'][gridpoints:] = gid+gridpoints

    
    if (dreduced==True):

        rho_out[rgid] = rho[:]
        u_out[rgid]   = u[:]
        w_out[rgid]   = w[:]
        e_out[rgid]   = e[:]

        amg_f.createVariable('rho', 'f8', ('no_of_points',))
        amg_f.variables['rho'][:gridpoints] = rho_out[:].real
        amg_f.variables['rho'][gridpoints:] = amg_f.variables['rho'][:gridpoints]

        amg_f.createVariable('u', 'f8', ('no_of_points',))
        amg_f.variables['u'][:gridpoints] = u_out[:].real
        amg_f.variables['u'][gridpoints:] = amg_f.variables['u'][:gridpoints]

        amg_f.createVariable('w', 'f8', ('no_of_points',))
        amg_f.variables['w'][:gridpoints] = w_out[:].real
        amg_f.variables['w'][gridpoints:] = amg_f.variables['w'][:gridpoints]

        amg_f.createVariable('e', 'f8', ('no_of_points',))
        amg_f.variables['e'][:gridpoints] = e_out[:].real
        amg_f.variables['e'][gridpoints:] = amg_f.variables['e'][:gridpoints]
        
        ######

        amg_f.createVariable('rho_i', 'f8', ('no_of_points',))
        amg_f.variables['rho_i'][:gridpoints] = rho_out[:].imag
        amg_f.variables['rho_i'][gridpoints:] = amg_f.variables['rho_i'][:gridpoints]

        amg_f.createVariable('u_i', 'f8', ('no_of_points',))
        amg_f.variables['u_i'][:gridpoints] = u_out[:].imag
        amg_f.variables['u_i'][gridpoints:] = amg_f.variables['u_i'][:gridpoints]

        amg_f.createVariable('w_i', 'f8', ('no_of_points',))
        amg_f.variables['w_i'][:gridpoints] = w_out[:].imag
        amg_f.variables['w_i'][gridpoints:] = amg_f.variables['w_i'][:gridpoints]

        amg_f.createVariable('e_i', 'f8', ('no_of_points',))
        amg_f.variables['e_i'][:gridpoints] = e_out[:].imag
        amg_f.variables['e_i'][gridpoints:] = amg_f.variables['e_i'][:gridpoints]
        
        if (neq==5):
            turb1_out[rgid] = turb1[:]
            amg_f.createVariable('turb1', 'f8', ('no_of_points',))
            amg_f.variables['turb1'][:gridpoints] = turb1_out[:].real
            amg_f.variables['turb1'][gridpoints:] = amg_f.variables['turb1'][:gridpoints]
            
            amg_f.createVariable('turb1_i', 'f8', ('no_of_points',))
            amg_f.variables['turb1_i'][:gridpoints] = turb1_out[:].imag
            amg_f.variables['turb1_i'][gridpoints:] = amg_f.variables['turb1_i'][:gridpoints]
        elif (neq==6):
            turb1_out[rgid] = turb1[:]
            turb2_out[rgid] = turb2[:]
           
            amg_f.createVariable('turb1', 'f8', ('no_of_points',))
            amg_f.variables['turb1'][:gridpoints] = turb1_out[:].real
            amg_f.variables['turb1'][gridpoints:] = amg_f.variables['turb1'][:gridpoints]
            
            amg_f.createVariable('turb1_i', 'f8', ('no_of_points',))
            amg_f.variables['turb1_i'][:gridpoints] = turb1_out[:].imag
            amg_f.variables['turb1_i'][gridpoints:] = amg_f.variables['turb1_i'][:gridpoints]
        
            amg_f.createVariable('turb2', 'f8', ('no_of_points',))
            amg_f.variables['turb2'][:gridpoints] = turb2_out[:].real
            amg_f.variables['turb2'][gridpoints:] = amg_f.variables['turb1'][:gridpoints]
            
            amg_f.createVariable('turb2_i', 'f8', ('no_of_points',))
            amg_f.variables['turb2_i'][:gridpoints] = turb2_out[:].imag
            amg_f.variables['turb2_i'][gridpoints:] = amg_f.variables['turb1_i'][:gridpoints]
    
    else:
        amg_f.createVariable('rho', 'f8', ('no_of_points',))
        amg_f.variables['rho'][:gridpoints] = rho[:].real
        amg_f.variables['rho'][gridpoints:] = amg_f.variables['rho'][:gridpoints]

        amg_f.createVariable('u', 'f8', ('no_of_points',))
        amg_f.variables['u'][:gridpoints] = u[:].real
        amg_f.variables['u'][gridpoints:] = amg_f.variables['u'][:gridpoints]

        amg_f.createVariable('w', 'f8', ('no_of_points',))
        amg_f.variables['w'][:gridpoints] = w[:].real
        amg_f.variables['w'][gridpoints:] = amg_f.variables['w'][:gridpoints]

        amg_f.createVariable('e', 'f8', ('no_of_points',))
        amg_f.variables['e'][:gridpoints] = e[:].real
        amg_f.variables['e'][gridpoints:] = amg_f.variables['e'][:gridpoints]
        
        ###

        amg_f.createVariable('rho_i', 'f8', ('no_of_points',))
        amg_f.variables['rho_i'][:gridpoints] = rho[:].imag
        amg_f.variables['rho_i'][gridpoints:] = amg_f.variables['rho_i'][:gridpoints]

        amg_f.createVariable('u_i', 'f8', ('no_of_points',))
        amg_f.variables['u_i'][:gridpoints] = u[:].imag
        amg_f.variables['u_i'][gridpoints:] = amg_f.variables['u_i'][:gridpoints]

        amg_f.createVariable('w_i', 'f8', ('no_of_points',))
        amg_f.variables['w_i'][:gridpoints] = w[:].imag
        amg_f.variables['w_i'][gridpoints:] = amg_f.variables['w_i'][:gridpoints]

        amg_f.createVariable('e_i', 'f8', ('no_of_points',))
        amg_f.variables['e_i'][:gridpoints] = e[:].imag
        amg_f.variables['e_i'][gridpoints:] = amg_f.variables['e_i'][:gridpoints]
        
        if (neq==5):
            amg_f.createVariable('turb1', 'f8', ('no_of_points',))
            amg_f.variables['turb1'][:gridpoints] = np.real(turb1[:])
            amg_f.variables['turb1'][gridpoints:] = amg_f.variables['turb1'][:gridpoints]
            
            amg_f.createVariable('turb1_i', 'f8', ('no_of_points',))
            amg_f.variables['turb1_i'][:gridpoints] = np.imag(turb1[:])
            amg_f.variables['turb1_i'][gridpoints:] = amg_f.variables['turb1_i'][:gridpoints]
        elif (neq==6):
            amg_f.createVariable('turb1', 'f8', ('no_of_points',))
            amg_f.variables['turb1'][:gridpoints] = np.real(turb1[:])
            amg_f.variables['turb1'][gridpoints:] = amg_f.variables['turb1'][:gridpoints]
            
            amg_f.createVariable('turb1_i', 'f8', ('no_of_points',))
            amg_f.variables['turb1_i'][:gridpoints] = np.imag(turb1[:])
            amg_f.variables['turb1_i'][gridpoints:] = amg_f.variables['turb1_i'][:gridpoints]
        
            amg_f.createVariable('turb2', 'f8', ('no_of_points',))
            amg_f.variables['turb2'][:gridpoints] = np.real(turb2[:])
            amg_f.variables['turb2'][gridpoints:] = amg_f.variables['turb1'][:gridpoints]
            
            amg_f.createVariable('turb2_i', 'f8', ('no_of_points',))
            amg_f.variables['turb2_i'][:gridpoints] = np.imag(turb2[:])
            amg_f.variables['turb2_i'][gridpoints:] = amg_f.variables['turb1_i'][:gridpoints]
    

    amg_f.close()

############################################################################

def mode2pval3D(filename, sol, npoints, nred, neq, beta, nums, dreduced=False, rgid=None):
    gridpoints = nred/neq
    rho = np.zeros(gridpoints, dtype='c16')
    u   = np.zeros(gridpoints, dtype='c16')
    v   = np.zeros(gridpoints, dtype='c16')
    w   = np.zeros(gridpoints, dtype='c16')
    e   = np.zeros(gridpoints, dtype='c16')

    if (beta==0):
        for i in xrange(0,gridpoints):
            rho[i] = sol[i*neq]
            u[i]   = sol[i*neq+1]
            w[i]   = sol[i*neq+2]
            e[i]   = sol[i*neq+3]
    else:
        for i in xrange(0,gridpoints):
            rho[i] = sol[i*neq]
            u[i]   = sol[i*neq+1]
            v[i]   = sol[i*neq+2]
            w[i]   = sol[i*neq+3]
            e[i]   = sol[i*neq+4]

    ## NETCDF SENSITIVITY OUTPUT ##
    amg_f = Dataset(filename[:-4]+'3D.pval', 'w', format="NETCDF3_64BIT_OFFSET")
    gridpoints = npoints/neq  ## SOLUTION FILE HAS ALL THE POINTS!
    amg_f.createDimension('no_of_points', gridpoints*nums)

    rho_out = np.zeros(gridpoints, dtype='c16')
    u_out   = np.zeros(gridpoints, dtype='c16')
    v_out   = np.zeros(gridpoints, dtype='c16')
    w_out   = np.zeros(gridpoints, dtype='c16')
    e_out   = np.zeros(gridpoints, dtype='c16')

    amg_f.createVariable('global_id', 'i', ('no_of_points',))
    amg_f.variables['global_id'][:] = np.arange(gridpoints*nums)

    if (dreduced==True):

        rho_out[rgid] = np.real(rho[:])
        u_out[rgid]   = np.real(u[:])
        v_out[rgid]   = np.real(v[:])
        w_out[rgid]   = np.real(w[:])
        e_out[rgid]   = np.real(e[:])

        amg_f.createVariable('rho', 'f8', ('no_of_points',))
        amg_f.variables['rho'][:gridpoints] = np.real(rho_out[:])

        amg_f.createVariable('u', 'f8', ('no_of_points',))
        amg_f.variables['u'][:gridpoints] = np.real(u_out[:])

        amg_f.createVariable('v', 'f8', ('no_of_points',))
        amg_f.variables['v'][:gridpoints] = np.real(v_out[:])

        amg_f.createVariable('w', 'f8', ('no_of_points',))
        amg_f.variables['w'][:gridpoints] = np.real(w_out[:])

        amg_f.createVariable('e', 'f8', ('no_of_points',))
        amg_f.variables['e'][:gridpoints] = np.real(e_out[:])
    else:
        amg_f.createVariable('rho', 'f8', ('no_of_points',))
        amg_f.variables['rho'][:gridpoints] = np.real(rho[:])

        amg_f.createVariable('u', 'f8', ('no_of_points',))
        amg_f.variables['u'][:gridpoints] = np.real(u[:])

        amg_f.createVariable('v', 'f8', ('no_of_points',))
        amg_f.variables['v'][:gridpoints] = np.real(v[:])

        amg_f.createVariable('w', 'f8', ('no_of_points',))
        amg_f.variables['w'][:gridpoints] = np.real(w[:])

        amg_f.createVariable('e', 'f8', ('no_of_points',))
        amg_f.variables['e'][:gridpoints] = np.real(e[:])

    variables = ['rho', 'u', 'w', 'e']
    Ly = 2*np.pi/nums
    for slice in range(0,nums):
        for var in variables:
            amg_f.variables[var][gridpoints*slice:gridpoints*(slice+1)] = amg_f.variables[var][:gridpoints]
        amg_f.variables['v'][gridpoints*slice:gridpoints*(slice+1)] = np.real( amg_f.variables['v'][:gridpoints]*np.exp(1j*Ly*slice) )
    
    amg_f.close()

##########################

    return
