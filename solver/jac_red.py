#! /usr/bin/env python

import sys
import numpy as np
from scipy.sparse import coo_matrix

class domain_reduction(object):
    """docstring for domain_reduction."""
    def __init__(self, zmin, zmax, xmin, xmax):
        super(domain_reduction, self).__init__()
        self.zmin = zmin
        self.zmax = zmax
        self.xmin = xmin
        self.xmax = xmax
        self.P = None
        self.PO = None
        self.POT = None
        self.m = None
        pass

    def create_Pmatrix(self, coords):
        """Generates the permutation matrix using the coordinates numpy array
           and the coordinates limits defined on the object construction."""
        nvar = coords.shape[0]
        self.P = coo_matrix((nvar,nvar))
        self.P.data = np.ones(nvar)
        self.P.row  = np.zeros(nvar, dtype='i4')
        self.P.col  = np.zeros(nvar, dtype='i4')

        ii = 0; jj = 1
        for i in range(nvar):
            if (coords[i,1] > self.zmin) and (coords[i,1] < self.zmax) \
                and (coords[i,0] > self.xmin) and (coords[i,0] < self.xmax):
                self.P.row[i] = ii
                self.P.col[i] = i
                ii += 1
            else:
                self.P.row[i] = nvar - jj
                self.P.col[i] = i
                jj += 1

        self.PO = self.P.tocsr()
        self.POT = self.P.tocsr().transpose()
        self.P = None
        self.m = ii
        print(' New dim = ', self.m, ' Current dim = ', nvar)
        print('')
        pass

    def reduce_matrix(self,A):
        """Reorder the matrix A with the permutation matrices and extracts a new
           reduced matrix C."""
        B = self.PO * A * self.POT
        C = B.tocsr()[0:self.m,:].tocsc()[:,0:self.m]
        B=None
        return C.tocsr()

    def reduce_vector(self,V):
        """Reorder a vector V and return only the reduced version of it."""
        newV = self.PO.dot(V)
        return newV[:self.m]
        
#def adapt_coord(coordfile):
#    with open(coordfile)
