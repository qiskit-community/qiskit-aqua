# -*- coding: utf-8 -*-

"""
This program is part of the PyQuante quantum chemistry program suite.

Copyright (c) 2004, Richard P. Muller. All Rights Reserved.

PyQuante version 1.2 and later is covered by the modified BSD
license. Please see the file LICENSE that is part of this
distribution.
"""

# =================================================================== #
# The two functions here, which are not provided by PyQuante 2, were  #
# copied from the PyQuante 1.6.5 suite.                               #
# =================================================================== #

import numpy as np


# From PyQuante file CI.py where it was "def TransformInts(Ints,orbs):"
#
def transformintegrals(integrals, orbs):
    """O(N^5) 4-index transformation of the two-electron integrals. Not as
    efficient as it could be, since it inflates to the full rectangular
    matrices rather than keeping them compressed. But at least it gets the
    correct result."""

    nbf,nmo = orbs.shape
    totlen = int(nmo*(nmo+1)*(nmo*nmo+nmo+2)/8)

    temp = np.zeros((nbf,nbf,nbf,nmo),'d')
    tempvec = np.zeros(nbf,'d')
    temp2 = np.zeros((nbf,nbf,nmo,nmo),'d')

    mos = range(nmo) # preform so we don't form inside loops
    bfs = range(nbf)

    # Start with (mu,nu|sigma,eta)
    # Unpack aoints and transform eta -> l
    for mu in bfs:
        for nu in bfs:
            for sigma in bfs:
                for l in mos:
                    for eta in bfs:
                        tempvec[eta] = integrals[mu,nu,sigma,eta]
                    temp[mu,nu,sigma,l] = np.dot(orbs[:,l],tempvec)

    # Transform sigma -> k
    for mu in bfs:
        for nu in bfs:
            for l in mos:
                for k in mos:
                    temp2[mu,nu,k,l] = np.dot(orbs[:,k],temp[mu,nu,:,l])

    # Transform nu -> j
    for mu in bfs:
        for k in mos:
            for l in mos:
                for j in mos:
                    temp[mu,j,k,l] = np.dot(orbs[:,j],temp2[mu,:,k,l])

    # Transform mu -> i and repack integrals:
    mointegrals = np.zeros(totlen,'d')
    for i in mos:
        for j in range(i+1):
            ij = i*(i+1)/2+j
            for k in mos:
                for l in range(k+1):
                    kl = k*(k+1)/2+l
                    if ij >= kl:
                        ijkl = ijkl2intindex(i,j,k,l)
                        mointegrals[ijkl] = np.dot(orbs[:,i],temp[:,j,k,l])

    del temp,temp2,tempvec #force garbage collection now
    return mointegrals


# From PyQuante file pyints.py
#
def ijkl2intindex(i,j,k,l):
    "Indexing into the get2ints long array"
    if i<j: i,j = j,i
    if k<l: k,l = l,k
    ij = i*(i+1)/2+j
    kl = k*(k+1)/2+l
    if ij < kl: ij,kl = kl,ij
    return int(ij*(ij+1)/2+kl)
