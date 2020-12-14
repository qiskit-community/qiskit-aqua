import numpy as np
import itertools
from qiskit.chemistry import FermionicOperator
from .analytic_qeom_fermionic_operator import FermionicOperatorNBody
from typing import List, Union, Optional, Tuple, Dict, cast


def delta(p, r):
    if(p == r):
        return 1
    return 0


def commutator_adj_nor(n: int, Emu: List, Enu: List):
    """
      Args:
      n [int]:    number of orbitals
      Emu [list]: excitation operator
      Enu [list]: excitation operator

      constructs the FermionicOperator representation of operator
      V_{IJ} = (Psi|[dag(E_I),E_J]|Psi)
    """

    hs = [None]*4
    idx = [None]*4
    if(len(Emu) == 2 and len(Enu) == 2):
        a, i = Emu
        b, j = Enu
        hs[0] = np.zeros(tuple([n]*2))
        idx[0] = []
        hs[0][i, j] += 1*delta(a, b)
        idx[0] += [(i, j)]
        hs[0][b, a] += -1*delta(j, i)
        idx[0] += [(b, a)]
    if(len(Emu) == 2 and len(Enu) == 4):
        a, i = Emu
        b, c, k, j = Enu
        hs[1] = np.zeros(tuple([n]*4))
        idx[1] = []
        hs[1][c, k, i, j] += 1*delta(a, b)
        idx[1] += [(c, k, i, j)]
        hs[1][b, k, i, j] += -1*delta(a, c)
        idx[1] += [(b, k, i, j)]
        hs[1][b, j, c, a] += -1*delta(k, i)
        idx[1] += [(b, j, c, a)]
        hs[1][b, k, c, a] += 1*delta(j, i)
        idx[1] += [(b, k, c, a)]
    if(len(Emu) == 4 and len(Enu) == 4):
        a, b, j, i = Emu
        c, d, l, k = Enu
        hs[1] = np.zeros(tuple([n]*4))
        idx[1] = []
        hs[2] = np.zeros(tuple([n]*6))
        idx[2] = []
        hs[1][c, b, d, a] += 1*delta(k, i)*delta(l, j)
        idx[1] += [(c, b, d, a)]
        hs[1][c, b, d, a] += -1*delta(l, i)*delta(k, j)
        idx[1] += [(c, b, d, a)]
        hs[1][i, l, j, k] += -1*delta(a, c)*delta(b, d)
        idx[1] += [(i, l, j, k)]
        hs[1][i, l, j, k] += 1*delta(b, c)*delta(a, d)
        idx[1] += [(i, l, j, k)]
        hs[2][d, l, i, k, j, a] += -1*delta(b, c)
        idx[2] += [(d, l, i, k, j, a)]
        hs[2][d, l, i, k, j, b] += 1*delta(a, c)
        idx[2] += [(d, l, i, k, j, b)]
        hs[2][c, k, d, b, j, a] += 1*delta(l, i)
        idx[2] += [(c, k, d, b, j, a)]
        hs[2][c, l, d, b, j, a] += -1*delta(k, i)
        idx[2] += [(c, l, d, b, j, a)]
        hs[2][c, k, d, b, i, a] += -1*delta(l, j)
        idx[2] += [(c, k, d, b, i, a)]
        hs[2][c, l, d, b, i, a] += 1*delta(k, j)
        idx[2] += [(c, l, d, b, i, a)]
        hs[2][c, l, i, k, j, a] += 1*delta(b, d)
        idx[2] += [(c, l, i, k, j, a)]
        hs[2][c, l, i, k, j, b] += -1*delta(a, d)
        idx[2] += [(c, l, i, k, j, b)]
    for k in range(4):
        if(idx[k] is not None):
            idx[k] = list(set(idx[k]))
    return FermionicOperatorNBody(hs), idx


def commutator_adj_adj(n: int, Emu: List, Enu: List):
    """
      Args:
      n [int]:    number of orbitals
      Emu [list]: excitation operator
      Enu [list]: excitation operator

      constructs the FermionicOperator representation of operator
      W_{IJ} = (Psi|[dag(E_I),dag(E_J)]|Psi)
    """
    hs = [None]*4
    idx = [None]*4
    if(len(Emu) == 2 and len(Enu) == 2):
        a, i = Emu
        b, j = Enu
    if(len(Emu) == 2 and len(Enu) == 4):
        a, i = Emu
        b, c, k, j = Enu
    if(len(Emu) == 4 and len(Enu) == 4):
        a, b, j, i = Emu
        c, d, l, k = Enu
    for k in range(4):
        if(idx[k] is not None):
            idx[k] = list(set(idx[k]))
    return FermionicOperatorNBody(hs), idx


def triple_commutator_adj_onebody_nor(n:int, Emu:List, Enu:List, H:float):
    """
      Args:
      n [int]:          number of orbitals
      Emu [list]:       excitation operator
      Enu [list]:       excitation operator
      H [rank-2 float]: matrix corresponding to H[i,j] \hat{a}^\dagger_i \hat{a}_j
      constructs the FermionicOperator representation of operator
      M_{IJ} = (Psi|[dag(E_I),H,E_J]|Psi)
    """
    hs = [None]*4
    idx = [None]*4
    if(len(Emu) == 2 and len(Enu) == 2):
        a, i = Emu
        b, j = Enu
        hs[0] = np.zeros(tuple([n]*2))
        idx[0] = []
        hs[0][i, j] += 2*H[a, b]
        idx[0] += [(i, j)]
        hs[0][b, :] += -1*H[a, :]*delta(j, i)
        idx[0] += [(b, u) for u in range(n)]
        hs[0][:, j] += -1*H[:, i]*delta(a, b)
        idx[0] += [(u, j) for u in range(n)]
        hs[0][b, a] += 2*H[j, i]
        idx[0] += [(b, a)]
        hs[0][:, a] += -1*H[:, b]*delta(j, i)
        idx[0] += [(u, a) for u in range(n)]
        hs[0][i, :] += -1*H[j, :]*delta(a, b)
        idx[0] += [(i, u) for u in range(n)]
    if(len(Emu) == 2 and len(Enu) == 4):
        a, i = Emu
        b, c, k, j = Enu
        hs[1] = np.zeros(tuple([n]*4))
        idx[1] = []
        hs[1][c, k, i, j] += 2*H[a, b]
        idx[1] += [(c, k, i, j)]
        hs[1][b, k, i, j] += -2*H[a, c]
        idx[1] += [(b, k, i, j)]
        hs[1][b, :, c, j] += 1*H[a, :]*delta(k, i)
        idx[1] += [(b, u, c, j) for u in range(n)]
        hs[1][b, :, c, k] += -1*H[a, :]*delta(j, i)
        idx[1] += [(b, u, c, k) for u in range(n)]
        hs[1][c, k, :, j] += -1*H[:, i]*delta(a, b)
        idx[1] += [(c, k, u, j) for u in range(n)]
        hs[1][b, k, :, j] += 1*H[:, i]*delta(a, c)
        idx[1] += [(b, k, u, j) for u in range(n)]
        hs[1][b, j, c, a] += 2*H[k, i]
        idx[1] += [(b, j, c, a)]
        hs[1][b, k, c, a] += -2*H[j, i]
        idx[1] += [(b, k, c, a)]
        hs[1][i, k, :, j] += 1*H[:, b]*delta(a, c)
        idx[1] += [(i, k, u, j) for u in range(n)]
        hs[1][c, k, :, a] += -1*H[:, b]*delta(j, i)
        idx[1] += [(c, k, u, a) for u in range(n)]
        hs[1][c, j, :, a] += 1*H[:, b]*delta(k, i)
        idx[1] += [(c, j, u, a) for u in range(n)]
        hs[1][i, k, :, j] += -1*H[:, c]*delta(a, b)
        idx[1] += [(i, k, u, j) for u in range(n)]
        hs[1][b, k, :, a] += 1*H[:, c]*delta(j, i)
        idx[1] += [(b, k, u, a) for u in range(n)]
        hs[1][b, j, :, a] += -1*H[:, c]*delta(k, i)
        idx[1] += [(b, j, u, a) for u in range(n)]
        hs[1][c, :, i, j] += -1*H[k, :]*delta(a, b)
        idx[1] += [(c, u, i, j) for u in range(n)]
        hs[1][b, :, i, j] += 1*H[k, :]*delta(a, c)
        idx[1] += [(b, u, i, j) for u in range(n)]
        hs[1][b, :, c, a] += -1*H[k, :]*delta(j, i)
        idx[1] += [(b, u, c, a) for u in range(n)]
        hs[1][c, :, i, k] += 1*H[j, :]*delta(a, b)
        idx[1] += [(c, u, i, k) for u in range(n)]
        hs[1][b, :, i, k] += -1*H[j, :]*delta(a, c)
        idx[1] += [(b, u, i, k) for u in range(n)]
        hs[1][b, :, c, a] += 1*H[j, :]*delta(k, i)
        idx[1] += [(b, u, c, a) for u in range(n)]
    if(len(Emu) == 4 and len(Enu) == 4):
        a, b, j, i = Emu
        c, d, l, k = Enu
        hs[1] = np.zeros(tuple([n]*4))
        idx[1] = []
        hs[2] = np.zeros(tuple([n]*6))
        idx[2] = []
        hs[1][i, :, j, l] += -1*H[k, :]*delta(a, c)*delta(b, d)
        idx[1] += [(i, u, j, l) for u in range(n)]
        hs[1][i, :, j, l] += 1*H[k, :]*delta(b, c)*delta(a, d)
        idx[1] += [(i, u, j, l) for u in range(n)]
        hs[1][i, :, j, k] += 1*H[l, :]*delta(a, c)*delta(b, d)
        idx[1] += [(i, u, j, k) for u in range(n)]
        hs[1][i, :, j, k] += -1*H[l, :]*delta(b, c)*delta(a, d)
        idx[1] += [(i, u, j, k) for u in range(n)]
        hs[1][c, b, :, a] += -1*H[:, d]*delta(l, i)*delta(k, j)
        idx[1] += [(c, b, u, a) for u in range(n)]
        hs[1][c, b, :, a] += 1*H[:, d]*delta(k, i)*delta(l, j)
        idx[1] += [(c, b, u, a) for u in range(n)]
        hs[1][d, b, :, a] += 1*H[:, c]*delta(l, i)*delta(k, j)
        idx[1] += [(d, b, u, a) for u in range(n)]
        hs[1][d, b, :, a] += -1*H[:, c]*delta(k, i)*delta(l, j)
        idx[1] += [(d, b, u, a) for u in range(n)]
        hs[1][c, b, d, a] += -2*H[l, j]*delta(k, i)
        idx[1] += [(c, b, d, a)]
        hs[1][c, b, d, a] += 2*H[k, j]*delta(l, i)
        idx[1] += [(c, b, d, a)]
        hs[1][i, l, :, k] += -1*H[:, j]*delta(b, c)*delta(a, d)
        idx[1] += [(i, l, u, k) for u in range(n)]
        hs[1][i, l, :, k] += 1*H[:, j]*delta(a, c)*delta(b, d)
        idx[1] += [(i, l, u, k) for u in range(n)]
        hs[1][c, b, d, a] += 2*H[l, i]*delta(k, j)
        idx[1] += [(c, b, d, a)]
        hs[1][c, b, d, a] += -2*H[k, i]*delta(l, j)
        idx[1] += [(c, b, d, a)]
        hs[1][j, l, :, k] += 1*H[:, i]*delta(b, c)*delta(a, d)
        idx[1] += [(j, l, u, k) for u in range(n)]
        hs[1][j, l, :, k] += -1*H[:, i]*delta(a, c)*delta(b, d)
        idx[1] += [(j, l, u, k) for u in range(n)]
        hs[1][c, :, d, b] += -1*H[a, :]*delta(k, i)*delta(l, j)
        idx[1] += [(c, u, d, b) for u in range(n)]
        hs[1][c, :, d, b] += 1*H[a, :]*delta(l, i)*delta(k, j)
        idx[1] += [(c, u, d, b) for u in range(n)]
        hs[1][i, l, j, k] += -2*H[a, c]*delta(b, d)
        idx[1] += [(i, l, j, k)]
        hs[1][i, l, j, k] += 2*H[a, d]*delta(b, c)
        idx[1] += [(i, l, j, k)]
        hs[1][c, :, d, a] += 1*H[b, :]*delta(k, i)*delta(l, j)
        idx[1] += [(c, u, d, a) for u in range(n)]
        hs[1][c, :, d, a] += -1*H[b, :]*delta(l, i)*delta(k, j)
        idx[1] += [(c, u, d, a) for u in range(n)]
        hs[1][i, l, j, k] += 2*H[b, c]*delta(a, d)
        idx[1] += [(i, l, j, k)]
        hs[1][i, l, j, k] += -2*H[b, d]*delta(a, c)
        idx[1] += [(i, l, j, k)]
        hs[2][d, l, j, k, :, b] += 1*H[:, i]*delta(a, c)
        idx[2] += [(d, l, j, k, u, b) for u in range(n)]
        hs[2][d, l, j, k, :, a] += -1*H[:, i]*delta(b, c)
        idx[2] += [(d, l, j, k, u, a) for u in range(n)]
        hs[2][c, l, j, k, :, b] += -1*H[:, i]*delta(a, d)
        idx[2] += [(c, l, j, k, u, b) for u in range(n)]
        hs[2][c, l, j, k, :, a] += 1*H[:, i]*delta(b, d)
        idx[2] += [(c, l, j, k, u, a) for u in range(n)]
        hs[2][c, :, d, k, i, a] += 1*H[b, :]*delta(l, j)
        idx[2] += [(c, u, d, k, i, a) for u in range(n)]
        hs[2][c, :, d, l, i, a] += -1*H[b, :]*delta(k, j)
        idx[2] += [(c, u, d, l, i, a) for u in range(n)]
        hs[2][c, k, d, b, :, a] += 1*H[:, i]*delta(l, j)
        idx[2] += [(c, k, d, b, u, a) for u in range(n)]
        hs[2][c, l, d, b, :, a] += -1*H[:, i]*delta(k, j)
        idx[2] += [(c, l, d, b, u, a) for u in range(n)]
        hs[2][c, k, d, b, j, a] += -2*H[l, i]
        idx[2] += [(c, k, d, b, j, a)]
        hs[2][c, l, d, b, j, a] += 2*H[k, i]
        idx[2] += [(c, l, d, b, j, a)]
        hs[2][c, :, i, l, j, k] += -1*H[b, :]*delta(a, d)
        idx[2] += [(c, u, i, l, j, k) for u in range(n)]
        hs[2][c, l, i, k, j, a] += 2*H[b, d]
        idx[2] += [(c, l, i, k, j, a)]
        hs[2][d, l, i, k, :, b] += -1*H[:, j]*delta(a, c)
        idx[2] += [(d, l, i, k, u, b) for u in range(n)]
        hs[2][d, l, i, k, :, a] += 1*H[:, j]*delta(b, c)
        idx[2] += [(d, l, i, k, u, a) for u in range(n)]
        hs[2][c, l, i, k, :, b] += 1*H[:, j]*delta(a, d)
        idx[2] += [(c, l, i, k, u, b) for u in range(n)]
        hs[2][c, l, i, k, :, a] += -1*H[:, j]*delta(b, d)
        idx[2] += [(c, l, i, k, u, a) for u in range(n)]
        hs[2][d, :, i, l, j, k] += -1*H[a, :]*delta(b, c)
        idx[2] += [(d, u, i, l, j, k) for u in range(n)]
        hs[2][d, l, i, k, j, b] += 2*H[a, c]
        idx[2] += [(d, l, i, k, j, b)]
        hs[2][c, k, d, b, :, a] += -1*H[:, j]*delta(l, i)
        idx[2] += [(c, k, d, b, u, a) for u in range(n)]
        hs[2][c, l, d, b, :, a] += 1*H[:, j]*delta(k, i)
        idx[2] += [(c, l, d, b, u, a) for u in range(n)]
        hs[2][c, k, d, b, i, a] += 2*H[l, j]
        idx[2] += [(c, k, d, b, i, a)]
        hs[2][c, l, d, b, i, a] += -2*H[k, j]
        idx[2] += [(c, l, d, b, i, a)]
        hs[2][c, :, i, l, j, k] += 1*H[a, :]*delta(b, d)
        idx[2] += [(c, u, i, l, j, k) for u in range(n)]
        hs[2][c, l, i, k, j, b] += -2*H[a, d]
        idx[2] += [(c, l, i, k, j, b)]
        hs[2][i, l, j, k, :, a] += 1*H[:, c]*delta(b, d)
        idx[2] += [(i, l, j, k, u, a) for u in range(n)]
        hs[2][i, l, j, k, :, b] += -1*H[:, c]*delta(a, d)
        idx[2] += [(i, l, j, k, u, b) for u in range(n)]
        hs[2][d, l, j, b, :, a] += -1*H[:, c]*delta(k, i)
        idx[2] += [(d, l, j, b, u, a) for u in range(n)]
        hs[2][d, k, j, b, :, a] += 1*H[:, c]*delta(l, i)
        idx[2] += [(d, k, j, b, u, a) for u in range(n)]
        hs[2][d, l, i, b, :, a] += 1*H[:, c]*delta(k, j)
        idx[2] += [(d, l, i, b, u, a) for u in range(n)]
        hs[2][d, k, i, b, :, a] += -1*H[:, c]*delta(l, j)
        idx[2] += [(d, k, i, b, u, a) for u in range(n)]
        hs[2][d, :, i, l, j, k] += 1*H[b, :]*delta(a, c)
        idx[2] += [(d, u, i, l, j, k) for u in range(n)]
        hs[2][d, l, i, k, j, a] += -2*H[b, c]
        idx[2] += [(d, l, i, k, j, a)]
        hs[2][i, l, j, k, :, a] += -1*H[:, d]*delta(b, c)
        idx[2] += [(i, l, j, k, u, a) for u in range(n)]
        hs[2][i, l, j, k, :, b] += 1*H[:, d]*delta(a, c)
        idx[2] += [(i, l, j, k, u, b) for u in range(n)]
        hs[2][c, l, j, b, :, a] += 1*H[:, d]*delta(k, i)
        idx[2] += [(c, l, j, b, u, a) for u in range(n)]
        hs[2][c, k, j, b, :, a] += -1*H[:, d]*delta(l, i)
        idx[2] += [(c, k, j, b, u, a) for u in range(n)]
        hs[2][c, l, i, b, :, a] += -1*H[:, d]*delta(k, j)
        idx[2] += [(c, l, i, b, u, a) for u in range(n)]
        hs[2][c, k, i, b, :, a] += 1*H[:, d]*delta(l, j)
        idx[2] += [(c, k, i, b, u, a) for u in range(n)]
        hs[2][c, :, d, k, j, b] += 1*H[a, :]*delta(l, i)
        idx[2] += [(c, u, d, k, j, b) for u in range(n)]
        hs[2][c, :, d, l, j, b] += -1*H[a, :]*delta(k, i)
        idx[2] += [(c, u, d, l, j, b) for u in range(n)]
        hs[2][d, :, i, k, j, a] += 1*H[l, :]*delta(b, c)
        idx[2] += [(d, u, i, k, j, a) for u in range(n)]
        hs[2][d, :, i, k, j, b] += -1*H[l, :]*delta(a, c)
        idx[2] += [(d, u, i, k, j, b) for u in range(n)]
        hs[2][c, :, i, k, j, a] += -1*H[l, :]*delta(b, d)
        idx[2] += [(c, u, i, k, j, a) for u in range(n)]
        hs[2][c, :, i, k, j, b] += 1*H[l, :]*delta(a, d)
        idx[2] += [(c, u, i, k, j, b) for u in range(n)]
        hs[2][c, :, d, k, i, b] += -1*H[a, :]*delta(l, j)
        idx[2] += [(c, u, d, k, i, b) for u in range(n)]
        hs[2][c, :, d, l, i, b] += 1*H[a, :]*delta(k, j)
        idx[2] += [(c, u, d, l, i, b) for u in range(n)]
        hs[2][c, :, d, b, j, a] += 1*H[l, :]*delta(k, i)
        idx[2] += [(c, u, d, b, j, a) for u in range(n)]
        hs[2][c, :, d, b, i, a] += -1*H[l, :]*delta(k, j)
        idx[2] += [(c, u, d, b, i, a) for u in range(n)]
        hs[2][d, :, i, l, j, a] += -1*H[k, :]*delta(b, c)
        idx[2] += [(d, u, i, l, j, a) for u in range(n)]
        hs[2][d, :, i, l, j, b] += 1*H[k, :]*delta(a, c)
        idx[2] += [(d, u, i, l, j, b) for u in range(n)]
        hs[2][c, :, i, l, j, a] += 1*H[k, :]*delta(b, d)
        idx[2] += [(c, u, i, l, j, a) for u in range(n)]
        hs[2][c, :, i, l, j, b] += -1*H[k, :]*delta(a, d)
        idx[2] += [(c, u, i, l, j, b) for u in range(n)]
        hs[2][c, :, d, k, j, a] += -1*H[b, :]*delta(l, i)
        idx[2] += [(c, u, d, k, j, a) for u in range(n)]
        hs[2][c, :, d, l, j, a] += 1*H[b, :]*delta(k, i)
        idx[2] += [(c, u, d, l, j, a) for u in range(n)]
        hs[2][c, :, d, b, j, a] += -1*H[k, :]*delta(l, i)
        idx[2] += [(c, u, d, b, j, a) for u in range(n)]
        hs[2][c, :, d, b, i, a] += 1*H[k, :]*delta(l, j)
        idx[2] += [(c, u, d, b, i, a) for u in range(n)]
    for k in range(4):
        if(idx[k] is not None):
            idx[k] = list(set(idx[k]))
    return FermionicOperatorNBody(hs), idx


def triple_commutator_adj_onebody_adj(n, Emu, Enu, H):
    """
      Args:
      n [int]:          number of orbitals
      Emu [list]:       excitation operator
      Enu [list]:       excitation operator
      H [rank-2 float]: matrix corresponding to H[i,j] \hat{a}^\dagger_i \hat{a}_j
      constructs the FermionicOperator representation of operator
      Q_{IJ} = (Psi|[dag(E_I),H,dag(E_J)]|Psi)
    """
    hs = [None]*4
    idx = [None]*4
    if(len(Emu) == 2 and len(Enu) == 2):
        a, i = Emu
        b, j = Enu
        hs[0] = np.zeros(tuple([n]*2))
        idx[0] = []
        hs[0][i, b] += 2*H[a, j]
        idx[0] += [(i, b)]
        hs[0][j, a] += 2*H[b, i]
        idx[0] += [(j, a)]
    if(len(Emu) == 2 and len(Enu) == 4):
        a, i = Emu
        b, c, k, j = Enu
        hs[1] = np.zeros(tuple([n]*4))
        idx[1] = []
        hs[1][i, c, k, b] += -2*H[a, j]
        idx[1] += [(i, c, k, b)]
        hs[1][i, c, j, b] += 2*H[a, k]
        idx[1] += [(i, c, j, b)]
        hs[1][j, b, k, a] += 2*H[c, i]
        idx[1] += [(j, b, k, a)]
        hs[1][j, c, k, a] += -2*H[b, i]
        idx[1] += [(j, c, k, a)]
    if(len(Emu) == 4 and len(Enu) == 4):
        a, b, j, i = Emu
        c, d, l, k = Enu
        hs[2] = np.zeros(tuple([n]*6))
        idx[2] = []
        hs[2][i, d, j, c, l, a] += -2*H[b, k]
        idx[2] += [(i, d, j, c, l, a)]
        hs[2][i, d, j, c, k, a] += 2*H[b, l]
        idx[2] += [(i, d, j, c, k, a)]
        hs[2][i, d, j, c, l, b] += 2*H[a, k]
        idx[2] += [(i, d, j, c, l, b)]
        hs[2][i, d, j, c, k, b] += -2*H[a, l]
        idx[2] += [(i, d, j, c, k, b)]
        hs[2][j, c, k, b, l, a] += -2*H[d, i]
        idx[2] += [(j, c, k, b, l, a)]
        hs[2][j, d, k, b, l, a] += 2*H[c, i]
        idx[2] += [(j, d, k, b, l, a)]
        hs[2][i, c, k, b, l, a] += 2*H[d, j]
        idx[2] += [(i, c, k, b, l, a)]
        hs[2][i, d, k, b, l, a] += -2*H[c, j]
        idx[2] += [(i, d, k, b, l, a)]
    for k in range(4):
        if(idx[k] is not None):
            idx[k] = list(set(idx[k]))
    return FermionicOperatorNBody(hs), idx


def triple_commutator_adj_twobody_nor(n, Emu, Enu, H):
    """
      Args:
      n [int]:          number of orbitals
      Emu [list]:       excitation operator
      Enu [list]:       excitation operator
      H [rank-2 float]: matrix corresponding to H[i,j,k,l] \hat{a}^\dagger_i \hat{a}^\dagger_j \hat{a}_k \hat{a}_l
      constructs the FermionicOperator representation of operator
      M_{IJ} = (Psi|[dag(E_I),H,E_J]|Psi)
    """
    hs = [None]*4
    idx = [None]*4
    if(len(Emu) == 2 and len(Enu) == 2):
        a, i = Emu
        b, j = Enu
        hs[1] = np.zeros(tuple([n]*4))
        idx[1] = []
        hs[1][i, :, :, j] += -2*H[a, b, :, :].transpose(1, 0)
        idx[1] += [(i, u, v, j) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[1][i, :, :, j] += 2*H[a, :, :, b]
        idx[1] += [(i, u, v, j) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[1][b, :, :, :] += 1*H[a, :, :, :].transpose(2, 1, 0)*delta(j, i)
        idx[1] += [(b, u, v, w) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[1][b, :, i, :] += -2*H[a, :, j, :].transpose(1, 0)
        idx[1] += [(b, u, i, v) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[1][i, :, :, j] += 2*H[:, b, a, :].transpose(1, 0)
        idx[1] += [(i, u, v, j) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[1][i, :, :, j] += -2*H[:, :, a, b].transpose(1, 0)
        idx[1] += [(i, u, v, j) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[1][b, :, :, :] += -1*H[:, :, a, :].transpose(2, 0, 1)*delta(j, i)
        idx[1] += [(b, u, v, w) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[1][b, :, i, :] += 2*H[j, :, a, :].transpose(1, 0)
        idx[1] += [(b, u, i, v) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[1][:, :, :, j] += -1*H[:, :, :, i]*delta(a, b)
        idx[1] += [(u, v, w, j) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[1][:, j, :, a] += -2*H[:, b, :, i]
        idx[1] += [(u, j, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[1][b, :, :, a] += 2*H[j, :, :, i]
        idx[1] += [(b, u, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[1][b, :, :, a] += -2*H[:, :, j, i].transpose(1, 0)
        idx[1] += [(b, u, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[1][:, :, :, j] += 1*H[:, i, :, :].transpose(0, 2, 1)*delta(a, b)
        idx[1] += [(u, v, w, j) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[1][:, j, :, a] += 2*H[:, i, :, b]
        idx[1] += [(u, j, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[1][b, :, :, a] += -2*H[j, i, :, :].transpose(1, 0)
        idx[1] += [(b, u, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[1][b, :, :, a] += 2*H[:, i, j, :].transpose(1, 0)
        idx[1] += [(b, u, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[1][:, :, :, a] += -1*H[:, :, :, b]*delta(j, i)
        idx[1] += [(u, v, w, a) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[1][:, :, :, a] += 1*H[:, b, :, :].transpose(0, 2, 1)*delta(j, i)
        idx[1] += [(u, v, w, a) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[1][i, :, :, :] += 1*H[j, :, :, :].transpose(2, 1, 0)*delta(a, b)
        idx[1] += [(i, u, v, w) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[1][i, :, :, :] += -1*H[:, :, j, :].transpose(2, 0, 1)*delta(a, b)
        idx[1] += [(i, u, v, w) for (u, v, w) in itertools.product(range(n), repeat=3)]
    if(len(Emu) == 2 and len(Enu) == 4):
        a, i = Emu
        b, c, k, j = Enu
        hs[1] = np.zeros(tuple([n]*4))
        idx[1] = []
        hs[2] = np.zeros(tuple([n]*6))
        idx[2] = []
        hs[1][b, :, i, :] += 1*H[j, :, k, :].transpose(1, 0)*delta(a, c)
        idx[1] += [(b, u, i, v) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[1][c, :, i, :] += -1*H[j, :, k, :].transpose(1, 0)*delta(a, b)
        idx[1] += [(c, u, i, v) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[1][b, :, i, :] += -1*H[k, :, j, :].transpose(1, 0)*delta(a, c)
        idx[1] += [(b, u, i, v) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[1][c, :, i, :] += 1*H[k, :, j, :].transpose(1, 0)*delta(a, b)
        idx[1] += [(c, u, i, v) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[1][:, j, :, a] += -1*H[:, b, :, c]*delta(k, i)
        idx[1] += [(u, j, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[1][:, k, :, a] += 1*H[:, b, :, c]*delta(j, i)
        idx[1] += [(u, k, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[1][:, j, :, a] += 1*H[:, c, :, b]*delta(k, i)
        idx[1] += [(u, j, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[1][:, k, :, a] += -1*H[:, c, :, b]*delta(j, i)
        idx[1] += [(u, k, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[1][b, :, c, a] += -2*H[j, i, k, :]
        idx[1] += [(b, u, c, a) for u in range(n)]
        hs[1][b, :, c, a] += 2*H[k, i, j, :]
        idx[1] += [(b, u, c, a) for u in range(n)]
        hs[1][:, k, :, j] += -1*H[:, i, :, b]*delta(a, c)
        idx[1] += [(u, k, v, j) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[1][:, k, :, j] += 1*H[:, i, :, c]*delta(a, b)
        idx[1] += [(u, k, v, j) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[1][b, :, c, a] += 2*H[j, :, k, i]
        idx[1] += [(b, u, c, a) for u in range(n)]
        hs[1][b, :, c, a] += -2*H[k, :, j, i]
        idx[1] += [(b, u, c, a) for u in range(n)]
        hs[1][:, k, :, j] += 1*H[:, b, :, i]*delta(a, c)
        idx[1] += [(u, k, v, j) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[1][:, k, :, j] += -1*H[:, c, :, i]*delta(a, b)
        idx[1] += [(u, k, v, j) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[1][b, :, c, :] += -1*H[k, :, a, :].transpose(1, 0)*delta(j, i)
        idx[1] += [(b, u, c, v) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[1][b, :, c, :] += 1*H[j, :, a, :].transpose(1, 0)*delta(k, i)
        idx[1] += [(b, u, c, v) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[1][i, k, :, j] += -2*H[:, c, a, b]
        idx[1] += [(i, k, u, j) for u in range(n)]
        hs[1][i, k, :, j] += 2*H[:, b, a, c]
        idx[1] += [(i, k, u, j) for u in range(n)]
        hs[1][b, :, c, :] += 1*H[a, :, k, :].transpose(1, 0)*delta(j, i)
        idx[1] += [(b, u, c, v) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[1][b, :, c, :] += -1*H[a, :, j, :].transpose(1, 0)*delta(k, i)
        idx[1] += [(b, u, c, v) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[1][i, k, :, j] += 2*H[a, c, :, b]
        idx[1] += [(i, k, u, j) for u in range(n)]
        hs[1][i, k, :, j] += -2*H[a, b, :, c]
        idx[1] += [(i, k, u, j) for u in range(n)]
        hs[2][c, :, :, k, :, j] += 1*H[:, :, :, i].transpose(1, 0, 2)*delta(a, b)
        idx[2] += [(c, u, v, k, w, j) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[2][c, k, :, j, :, a] += -2*H[:, b, :, i]
        idx[2] += [(c, k, u, j, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][b, :, :, k, :, j] += -1*H[:, :, :, i].transpose(1, 0, 2)*delta(a, c)
        idx[2] += [(b, u, v, k, w, j) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[2][b, k, :, j, :, a] += 2*H[:, c, :, i]
        idx[2] += [(b, k, u, j, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][b, :, c, :, i, j] += -2*H[a, :, k, :].transpose(1, 0)
        idx[2] += [(b, u, c, v, i, j) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][b, :, c, :, i, k] += 2*H[a, :, j, :].transpose(1, 0)
        idx[2] += [(b, u, c, v, i, k) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][b, :, c, j, :, a] += -2*H[k, :, :, i]
        idx[2] += [(b, u, c, j, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][b, :, c, k, :, a] += 2*H[j, :, :, i]
        idx[2] += [(b, u, c, k, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][b, :, c, j, :, a] += 2*H[:, :, k, i].transpose(1, 0)
        idx[2] += [(b, u, c, j, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][b, :, c, k, :, a] += -2*H[:, :, j, i].transpose(1, 0)
        idx[2] += [(b, u, c, k, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][b, :, i, k, :, j] += -2*H[a, c, :, :].transpose(1, 0)
        idx[2] += [(b, u, i, k, v, j) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][b, :, i, k, :, j] += 2*H[a, :, :, c]
        idx[2] += [(b, u, i, k, v, j) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][c, :, :, k, :, j] += -1*H[:, i, :, :].transpose(2, 0, 1)*delta(a, b)
        idx[2] += [(c, u, v, k, w, j) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[2][c, k, :, j, :, a] += 2*H[:, i, :, b]
        idx[2] += [(c, k, u, j, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][b, :, :, k, :, j] += 1*H[:, i, :, :].transpose(2, 0, 1)*delta(a, c)
        idx[2] += [(b, u, v, k, w, j) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[2][b, k, :, j, :, a] += -2*H[:, i, :, c]
        idx[2] += [(b, k, u, j, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][c, :, i, k, :, j] += -2*H[:, b, a, :].transpose(1, 0)
        idx[2] += [(c, u, i, k, v, j) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][c, :, i, k, :, j] += 2*H[:, :, a, b].transpose(1, 0)
        idx[2] += [(c, u, i, k, v, j) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][b, :, c, j, :, a] += 2*H[k, i, :, :].transpose(1, 0)
        idx[2] += [(b, u, c, j, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][b, :, c, k, :, a] += -2*H[j, i, :, :].transpose(1, 0)
        idx[2] += [(b, u, c, k, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][b, :, c, j, :, a] += -2*H[:, i, k, :].transpose(1, 0)
        idx[2] += [(b, u, c, j, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][b, :, c, k, :, a] += 2*H[:, i, j, :].transpose(1, 0)
        idx[2] += [(b, u, c, k, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][b, :, i, k, :, j] += 2*H[:, c, a, :].transpose(1, 0)
        idx[2] += [(b, u, i, k, v, j) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][b, :, i, k, :, j] += -2*H[:, :, a, c].transpose(1, 0)
        idx[2] += [(b, u, i, k, v, j) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][i, :, :, k, :, j] += -1*H[:, :, :, b].transpose(1, 0, 2)*delta(a, c)
        idx[2] += [(i, u, v, k, w, j) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[2][c, :, :, k, :, a] += 1*H[:, :, :, b].transpose(1, 0, 2)*delta(j, i)
        idx[2] += [(c, u, v, k, w, a) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[2][c, :, :, j, :, a] += -1*H[:, :, :, b].transpose(1, 0, 2)*delta(k, i)
        idx[2] += [(c, u, v, j, w, a) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[2][i, :, :, k, :, j] += 1*H[:, b, :, :].transpose(2, 0, 1)*delta(a, c)
        idx[2] += [(i, u, v, k, w, j) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[2][c, :, :, k, :, a] += -1*H[:, b, :, :].transpose(2, 0, 1)*delta(j, i)
        idx[2] += [(c, u, v, k, w, a) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[2][c, :, :, j, :, a] += 1*H[:, b, :, :].transpose(2, 0, 1)*delta(k, i)
        idx[2] += [(c, u, v, j, w, a) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[2][i, :, :, k, :, j] += 1*H[:, :, :, c].transpose(1, 0, 2)*delta(a, b)
        idx[2] += [(i, u, v, k, w, j) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[2][b, :, :, k, :, a] += -1*H[:, :, :, c].transpose(1, 0, 2)*delta(j, i)
        idx[2] += [(b, u, v, k, w, a) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[2][b, :, :, j, :, a] += 1*H[:, :, :, c].transpose(1, 0, 2)*delta(k, i)
        idx[2] += [(b, u, v, j, w, a) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[2][i, :, :, k, :, j] += -1*H[:, c, :, :].transpose(2, 0, 1)*delta(a, b)
        idx[2] += [(i, u, v, k, w, j) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[2][b, :, :, k, :, a] += 1*H[:, c, :, :].transpose(2, 0, 1)*delta(j, i)
        idx[2] += [(b, u, v, k, w, a) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[2][b, :, :, j, :, a] += -1*H[:, c, :, :].transpose(2, 0, 1)*delta(k, i)
        idx[2] += [(b, u, v, j, w, a) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[2][c, :, i, k, :, j] += 2*H[a, b, :, :].transpose(1, 0)
        idx[2] += [(c, u, i, k, v, j) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][c, :, i, k, :, j] += -2*H[a, :, :, b]
        idx[2] += [(c, u, i, k, v, j) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][b, :, c, :, :, j] += -1*H[:, :, a, :].transpose(2, 1, 0)*delta(k, i)
        idx[2] += [(b, u, c, v, w, j) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[2][b, :, c, :, :, k] += 1*H[:, :, a, :].transpose(2, 1, 0)*delta(j, i)
        idx[2] += [(b, u, c, v, w, k) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[2][c, :, i, :, :, j] += -1*H[k, :, :, :].transpose(2, 0, 1)*delta(a, b)
        idx[2] += [(c, u, i, v, w, j) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[2][b, :, i, :, :, j] += 1*H[k, :, :, :].transpose(2, 0, 1)*delta(a, c)
        idx[2] += [(b, u, i, v, w, j) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[2][b, :, c, :, :, a] += -1*H[k, :, :, :].transpose(2, 0, 1)*delta(j, i)
        idx[2] += [(b, u, c, v, w, a) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[2][c, :, i, :, :, k] += 1*H[j, :, :, :].transpose(2, 0, 1)*delta(a, b)
        idx[2] += [(c, u, i, v, w, k) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[2][b, :, i, :, :, k] += -1*H[j, :, :, :].transpose(2, 0, 1)*delta(a, c)
        idx[2] += [(b, u, i, v, w, k) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[2][b, :, c, :, :, a] += 1*H[j, :, :, :].transpose(2, 0, 1)*delta(k, i)
        idx[2] += [(b, u, c, v, w, a) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[2][c, :, i, :, :, j] += 1*H[:, :, k, :].transpose(2, 1, 0)*delta(a, b)
        idx[2] += [(c, u, i, v, w, j) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[2][b, :, i, :, :, j] += -1*H[:, :, k, :].transpose(2, 1, 0)*delta(a, c)
        idx[2] += [(b, u, i, v, w, j) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[2][b, :, c, :, :, a] += 1*H[:, :, k, :].transpose(2, 1, 0)*delta(j, i)
        idx[2] += [(b, u, c, v, w, a) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[2][c, :, i, :, :, k] += -1*H[:, :, j, :].transpose(2, 1, 0)*delta(a, b)
        idx[2] += [(c, u, i, v, w, k) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[2][b, :, i, :, :, k] += 1*H[:, :, j, :].transpose(2, 1, 0)*delta(a, c)
        idx[2] += [(b, u, i, v, w, k) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[2][b, :, c, :, :, a] += -1*H[:, :, j, :].transpose(2, 1, 0)*delta(k, i)
        idx[2] += [(b, u, c, v, w, a) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[2][b, :, c, :, i, j] += 2*H[k, :, a, :].transpose(1, 0)
        idx[2] += [(b, u, c, v, i, j) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][b, :, c, :, i, k] += -2*H[j, :, a, :].transpose(1, 0)
        idx[2] += [(b, u, c, v, i, k) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][b, :, c, :, :, j] += 1*H[a, :, :, :].transpose(2, 0, 1)*delta(k, i)
        idx[2] += [(b, u, c, v, w, j) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[2][b, :, c, :, :, k] += -1*H[a, :, :, :].transpose(2, 0, 1)*delta(j, i)
        idx[2] += [(b, u, c, v, w, k) for (u, v, w) in itertools.product(range(n), repeat=3)]
    if(len(Emu) == 4 and len(Enu) == 4):
        a, b, j, i = Emu
        c, d, l, k = Enu
        hs[1] = np.zeros(tuple([n]*4))
        idx[1] = []
        hs[2] = np.zeros(tuple([n]*6))
        idx[2] = []
        hs[3] = np.zeros(tuple([n]*8))
        idx[3] = []
        hs[1][i, :, j, :] += 1*H[k, :, l, :].transpose(1, 0)*delta(a, c)*delta(b, d)
        idx[1] += [(i, u, j, v) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[1][i, :, j, :] += -1*H[k, :, l, :].transpose(1, 0)*delta(b, c)*delta(a, d)
        idx[1] += [(i, u, j, v) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][c, :, i, :, j, b] += 1*H[k, :, l, :].transpose(1, 0)*delta(a, d)
        idx[2] += [(c, u, i, v, j, b) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][c, :, i, :, j, a] += -1*H[k, :, l, :].transpose(1, 0)*delta(b, d)
        idx[2] += [(c, u, i, v, j, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][d, :, i, :, j, b] += -1*H[k, :, l, :].transpose(1, 0)*delta(a, c)
        idx[2] += [(d, u, i, v, j, b) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][d, :, i, :, j, a] += 1*H[k, :, l, :].transpose(1, 0)*delta(b, c)
        idx[2] += [(d, u, i, v, j, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[1][i, :, j, :] += -1*H[l, :, k, :].transpose(1, 0)*delta(a, c)*delta(b, d)
        idx[1] += [(i, u, j, v) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[1][i, :, j, :] += 1*H[l, :, k, :].transpose(1, 0)*delta(b, c)*delta(a, d)
        idx[1] += [(i, u, j, v) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][c, :, i, :, j, b] += -1*H[l, :, k, :].transpose(1, 0)*delta(a, d)
        idx[2] += [(c, u, i, v, j, b) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][c, :, i, :, j, a] += 1*H[l, :, k, :].transpose(1, 0)*delta(b, d)
        idx[2] += [(c, u, i, v, j, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][d, :, i, :, j, b] += 1*H[l, :, k, :].transpose(1, 0)*delta(a, c)
        idx[2] += [(d, u, i, v, j, b) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][d, :, i, :, j, a] += -1*H[l, :, k, :].transpose(1, 0)*delta(b, c)
        idx[2] += [(d, u, i, v, j, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][c, :, i, :, j, l] += 1*H[b, :, k, :].transpose(1, 0)*delta(a, d)
        idx[2] += [(c, u, i, v, j, l) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][c, :, i, :, j, l] += -1*H[a, :, k, :].transpose(1, 0)*delta(b, d)
        idx[2] += [(c, u, i, v, j, l) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][d, :, i, :, j, l] += -1*H[b, :, k, :].transpose(1, 0)*delta(a, c)
        idx[2] += [(d, u, i, v, j, l) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][d, :, i, :, j, l] += 1*H[a, :, k, :].transpose(1, 0)*delta(b, c)
        idx[2] += [(d, u, i, v, j, l) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][i, :, j, :, :, l] += 1*H[:, :, k, :].transpose(2, 1, 0)*delta(a, c)*delta(b, d)
        idx[2] += [(i, u, j, v, w, l) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[2][i, :, j, :, :, l] += -1*H[:, :, k, :].transpose(2, 1, 0)*delta(b, c)*delta(a, d)
        idx[2] += [(i, u, j, v, w, l) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[2][c, :, i, :, j, k] += -1*H[b, :, l, :].transpose(1, 0)*delta(a, d)
        idx[2] += [(c, u, i, v, j, k) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][c, :, i, :, j, k] += 1*H[a, :, l, :].transpose(1, 0)*delta(b, d)
        idx[2] += [(c, u, i, v, j, k) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][d, :, i, :, j, k] += 1*H[b, :, l, :].transpose(1, 0)*delta(a, c)
        idx[2] += [(d, u, i, v, j, k) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][d, :, i, :, j, k] += -1*H[a, :, l, :].transpose(1, 0)*delta(b, c)
        idx[2] += [(d, u, i, v, j, k) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][i, :, j, :, :, k] += -1*H[:, :, l, :].transpose(2, 1, 0)*delta(a, c)*delta(b, d)
        idx[2] += [(i, u, j, v, w, k) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[2][i, :, j, :, :, k] += 1*H[:, :, l, :].transpose(2, 1, 0)*delta(b, c)*delta(a, d)
        idx[2] += [(i, u, j, v, w, k) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[2][c, :, i, :, j, l] += -1*H[k, :, b, :].transpose(1, 0)*delta(a, d)
        idx[2] += [(c, u, i, v, j, l) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][c, :, i, :, j, l] += 1*H[k, :, a, :].transpose(1, 0)*delta(b, d)
        idx[2] += [(c, u, i, v, j, l) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][d, :, i, :, j, l] += 1*H[k, :, b, :].transpose(1, 0)*delta(a, c)
        idx[2] += [(d, u, i, v, j, l) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][d, :, i, :, j, l] += -1*H[k, :, a, :].transpose(1, 0)*delta(b, c)
        idx[2] += [(d, u, i, v, j, l) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][i, :, j, :, :, l] += -1*H[k, :, :, :].transpose(2, 0, 1)*delta(a, c)*delta(b, d)
        idx[2] += [(i, u, j, v, w, l) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[2][i, :, j, :, :, l] += 1*H[k, :, :, :].transpose(2, 0, 1)*delta(b, c)*delta(a, d)
        idx[2] += [(i, u, j, v, w, l) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[2][c, :, i, :, j, k] += 1*H[l, :, b, :].transpose(1, 0)*delta(a, d)
        idx[2] += [(c, u, i, v, j, k) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][c, :, i, :, j, k] += -1*H[l, :, a, :].transpose(1, 0)*delta(b, d)
        idx[2] += [(c, u, i, v, j, k) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][d, :, i, :, j, k] += -1*H[l, :, b, :].transpose(1, 0)*delta(a, c)
        idx[2] += [(d, u, i, v, j, k) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][d, :, i, :, j, k] += 1*H[l, :, a, :].transpose(1, 0)*delta(b, c)
        idx[2] += [(d, u, i, v, j, k) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][i, :, j, :, :, k] += 1*H[l, :, :, :].transpose(2, 0, 1)*delta(a, c)*delta(b, d)
        idx[2] += [(i, u, j, v, w, k) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[2][i, :, j, :, :, k] += -1*H[l, :, :, :].transpose(2, 0, 1)*delta(b, c)*delta(a, d)
        idx[2] += [(i, u, j, v, w, k) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[1][:, b, :, a] += -1*H[:, c, :, d]*delta(l, i)*delta(k, j)
        idx[1] += [(u, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[1][:, b, :, a] += 1*H[:, c, :, d]*delta(k, i)*delta(l, j)
        idx[1] += [(u, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][i, k, :, b, :, a] += -1*H[:, c, :, d]*delta(l, j)
        idx[2] += [(i, k, u, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][i, l, :, b, :, a] += 1*H[:, c, :, d]*delta(k, j)
        idx[2] += [(i, l, u, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][j, k, :, b, :, a] += 1*H[:, c, :, d]*delta(l, i)
        idx[2] += [(j, k, u, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][j, l, :, b, :, a] += -1*H[:, c, :, d]*delta(k, i)
        idx[2] += [(j, l, u, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[1][:, b, :, a] += 1*H[:, d, :, c]*delta(l, i)*delta(k, j)
        idx[1] += [(u, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[1][:, b, :, a] += -1*H[:, d, :, c]*delta(k, i)*delta(l, j)
        idx[1] += [(u, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][i, k, :, b, :, a] += 1*H[:, d, :, c]*delta(l, j)
        idx[2] += [(i, k, u, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][i, l, :, b, :, a] += -1*H[:, d, :, c]*delta(k, j)
        idx[2] += [(i, l, u, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][j, k, :, b, :, a] += -1*H[:, d, :, c]*delta(l, i)
        idx[2] += [(j, k, u, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][j, l, :, b, :, a] += 1*H[:, d, :, c]*delta(k, i)
        idx[2] += [(j, l, u, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][c, k, :, b, :, a] += -1*H[:, d, :, i]*delta(l, j)
        idx[2] += [(c, k, u, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][c, k, :, b, :, a] += 1*H[:, d, :, j]*delta(l, i)
        idx[2] += [(c, k, u, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][c, l, :, b, :, a] += 1*H[:, d, :, i]*delta(k, j)
        idx[2] += [(c, l, u, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][c, l, :, b, :, a] += -1*H[:, d, :, j]*delta(k, i)
        idx[2] += [(c, l, u, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][c, :, :, b, :, a] += -1*H[:, d, :, :].transpose(2, 0, 1)*delta(l, i)*delta(k, j)
        idx[2] += [(c, u, v, b, w, a) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[2][c, :, :, b, :, a] += 1*H[:, d, :, :].transpose(2, 0, 1)*delta(k, i)*delta(l, j)
        idx[2] += [(c, u, v, b, w, a) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[2][c, k, :, b, :, a] += 1*H[:, i, :, d]*delta(l, j)
        idx[2] += [(c, k, u, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][c, k, :, b, :, a] += -1*H[:, j, :, d]*delta(l, i)
        idx[2] += [(c, k, u, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][c, l, :, b, :, a] += -1*H[:, i, :, d]*delta(k, j)
        idx[2] += [(c, l, u, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][c, l, :, b, :, a] += 1*H[:, j, :, d]*delta(k, i)
        idx[2] += [(c, l, u, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][c, :, :, b, :, a] += 1*H[:, :, :, d].transpose(1, 0, 2)*delta(l, i)*delta(k, j)
        idx[2] += [(c, u, v, b, w, a) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[2][c, :, :, b, :, a] += -1*H[:, :, :, d].transpose(1, 0, 2)*delta(k, i)*delta(l, j)
        idx[2] += [(c, u, v, b, w, a) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[2][d, k, :, b, :, a] += 1*H[:, c, :, i]*delta(l, j)
        idx[2] += [(d, k, u, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][d, k, :, b, :, a] += -1*H[:, c, :, j]*delta(l, i)
        idx[2] += [(d, k, u, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][d, l, :, b, :, a] += -1*H[:, c, :, i]*delta(k, j)
        idx[2] += [(d, l, u, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][d, l, :, b, :, a] += 1*H[:, c, :, j]*delta(k, i)
        idx[2] += [(d, l, u, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][d, :, :, b, :, a] += 1*H[:, c, :, :].transpose(2, 0, 1)*delta(l, i)*delta(k, j)
        idx[2] += [(d, u, v, b, w, a) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[2][d, :, :, b, :, a] += -1*H[:, c, :, :].transpose(2, 0, 1)*delta(k, i)*delta(l, j)
        idx[2] += [(d, u, v, b, w, a) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[2][d, k, :, b, :, a] += -1*H[:, i, :, c]*delta(l, j)
        idx[2] += [(d, k, u, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][d, k, :, b, :, a] += 1*H[:, j, :, c]*delta(l, i)
        idx[2] += [(d, k, u, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][d, l, :, b, :, a] += 1*H[:, i, :, c]*delta(k, j)
        idx[2] += [(d, l, u, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][d, l, :, b, :, a] += -1*H[:, j, :, c]*delta(k, i)
        idx[2] += [(d, l, u, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][d, :, :, b, :, a] += -1*H[:, :, :, c].transpose(1, 0, 2)*delta(l, i)*delta(k, j)
        idx[2] += [(d, u, v, b, w, a) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[2][d, :, :, b, :, a] += 1*H[:, :, :, c].transpose(1, 0, 2)*delta(k, i)*delta(l, j)
        idx[2] += [(d, u, v, b, w, a) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[1][c, b, d, a] += -2*H[k, i, l, j]
        idx[1] += [(c, b, d, a)]
        hs[1][c, b, d, a] += 2*H[l, i, k, j]
        idx[1] += [(c, b, d, a)]
        hs[2][c, l, d, b, :, a] += -2*H[:, i, k, j]
        idx[2] += [(c, l, d, b, u, a) for u in range(n)]
        hs[2][c, k, d, b, :, a] += 2*H[:, i, l, j]
        idx[2] += [(c, k, d, b, u, a) for u in range(n)]
        hs[2][c, l, d, b, :, a] += 2*H[k, i, :, j]
        idx[2] += [(c, l, d, b, u, a) for u in range(n)]
        hs[2][c, k, d, b, :, a] += -2*H[l, i, :, j]
        idx[2] += [(c, k, d, b, u, a) for u in range(n)]
        hs[1][:, l, :, k] += -1*H[:, i, :, j]*delta(b, c)*delta(a, d)
        idx[1] += [(u, l, v, k) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[1][:, l, :, k] += 1*H[:, i, :, j]*delta(a, c)*delta(b, d)
        idx[1] += [(u, l, v, k) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][c, l, :, k, :, a] += -1*H[:, i, :, j]*delta(b, d)
        idx[2] += [(c, l, u, k, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][c, l, :, k, :, b] += 1*H[:, i, :, j]*delta(a, d)
        idx[2] += [(c, l, u, k, v, b) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][d, l, :, k, :, a] += 1*H[:, i, :, j]*delta(b, c)
        idx[2] += [(d, l, u, k, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][d, l, :, k, :, b] += -1*H[:, i, :, j]*delta(a, c)
        idx[2] += [(d, l, u, k, v, b) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[1][c, b, d, a] += 2*H[k, j, l, i]
        idx[1] += [(c, b, d, a)]
        hs[1][c, b, d, a] += -2*H[l, j, k, i]
        idx[1] += [(c, b, d, a)]
        hs[2][c, l, d, b, :, a] += 2*H[:, j, k, i]
        idx[2] += [(c, l, d, b, u, a) for u in range(n)]
        hs[2][c, k, d, b, :, a] += -2*H[:, j, l, i]
        idx[2] += [(c, k, d, b, u, a) for u in range(n)]
        hs[2][c, l, d, b, :, a] += -2*H[k, j, :, i]
        idx[2] += [(c, l, d, b, u, a) for u in range(n)]
        hs[2][c, k, d, b, :, a] += 2*H[l, j, :, i]
        idx[2] += [(c, k, d, b, u, a) for u in range(n)]
        hs[1][:, l, :, k] += 1*H[:, j, :, i]*delta(b, c)*delta(a, d)
        idx[1] += [(u, l, v, k) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[1][:, l, :, k] += -1*H[:, j, :, i]*delta(a, c)*delta(b, d)
        idx[1] += [(u, l, v, k) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][c, l, :, k, :, a] += 1*H[:, j, :, i]*delta(b, d)
        idx[2] += [(c, l, u, k, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][c, l, :, k, :, b] += -1*H[:, j, :, i]*delta(a, d)
        idx[2] += [(c, l, u, k, v, b) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][d, l, :, k, :, a] += -1*H[:, j, :, i]*delta(b, c)
        idx[2] += [(d, l, u, k, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][d, l, :, k, :, b] += 1*H[:, j, :, i]*delta(a, c)
        idx[2] += [(d, l, u, k, v, b) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][c, :, d, b, i, a] += -2*H[k, j, l, :]
        idx[2] += [(c, u, d, b, i, a) for u in range(n)]
        hs[2][c, :, d, b, i, a] += 2*H[l, j, k, :]
        idx[2] += [(c, u, d, b, i, a) for u in range(n)]
        hs[2][c, :, d, b, :, a] += 2*H[:, j, l, :].transpose(1, 0)*delta(k, i)
        idx[2] += [(c, u, d, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][c, :, d, b, :, a] += -2*H[:, j, k, :].transpose(1, 0)*delta(l, i)
        idx[2] += [(c, u, d, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][c, :, d, b, :, a] += -2*H[l, j, :, :].transpose(1, 0)*delta(k, i)
        idx[2] += [(c, u, d, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][c, :, d, b, :, a] += 2*H[k, j, :, :].transpose(1, 0)*delta(l, i)
        idx[2] += [(c, u, d, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][i, l, :, k, :, a] += -1*H[:, j, :, c]*delta(b, d)
        idx[2] += [(i, l, u, k, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][i, l, :, k, :, a] += 1*H[:, j, :, d]*delta(b, c)
        idx[2] += [(i, l, u, k, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][i, l, :, k, :, b] += 1*H[:, j, :, c]*delta(a, d)
        idx[2] += [(i, l, u, k, v, b) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][i, l, :, k, :, b] += -1*H[:, j, :, d]*delta(a, c)
        idx[2] += [(i, l, u, k, v, b) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][i, :, :, l, :, k] += -1*H[:, j, :, :].transpose(2, 0, 1)*delta(b, c)*delta(a, d)
        idx[2] += [(i, u, v, l, w, k) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[2][i, :, :, l, :, k] += 1*H[:, j, :, :].transpose(2, 0, 1)*delta(a, c)*delta(b, d)
        idx[2] += [(i, u, v, l, w, k) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[2][c, :, d, b, i, a] += 2*H[k, :, l, j]
        idx[2] += [(c, u, d, b, i, a) for u in range(n)]
        hs[2][c, :, d, b, i, a] += -2*H[l, :, k, j]
        idx[2] += [(c, u, d, b, i, a) for u in range(n)]
        hs[2][c, :, d, b, :, a] += -2*H[:, :, l, j].transpose(1, 0)*delta(k, i)
        idx[2] += [(c, u, d, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][c, :, d, b, :, a] += 2*H[:, :, k, j].transpose(1, 0)*delta(l, i)
        idx[2] += [(c, u, d, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][c, :, d, b, :, a] += 2*H[l, :, :, j]*delta(k, i)
        idx[2] += [(c, u, d, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][c, :, d, b, :, a] += -2*H[k, :, :, j]*delta(l, i)
        idx[2] += [(c, u, d, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][i, l, :, k, :, a] += 1*H[:, c, :, j]*delta(b, d)
        idx[2] += [(i, l, u, k, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][i, l, :, k, :, a] += -1*H[:, d, :, j]*delta(b, c)
        idx[2] += [(i, l, u, k, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][i, l, :, k, :, b] += -1*H[:, c, :, j]*delta(a, d)
        idx[2] += [(i, l, u, k, v, b) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][i, l, :, k, :, b] += 1*H[:, d, :, j]*delta(a, c)
        idx[2] += [(i, l, u, k, v, b) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][i, :, :, l, :, k] += 1*H[:, :, :, j].transpose(1, 0, 2)*delta(b, c)*delta(a, d)
        idx[2] += [(i, u, v, l, w, k) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[2][i, :, :, l, :, k] += -1*H[:, :, :, j].transpose(1, 0, 2)*delta(a, c)*delta(b, d)
        idx[2] += [(i, u, v, l, w, k) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[2][c, :, d, b, j, a] += 2*H[k, i, l, :]
        idx[2] += [(c, u, d, b, j, a) for u in range(n)]
        hs[2][c, :, d, b, j, a] += -2*H[l, i, k, :]
        idx[2] += [(c, u, d, b, j, a) for u in range(n)]
        hs[2][c, :, d, b, :, a] += -2*H[:, i, l, :].transpose(1, 0)*delta(k, j)
        idx[2] += [(c, u, d, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][c, :, d, b, :, a] += 2*H[:, i, k, :].transpose(1, 0)*delta(l, j)
        idx[2] += [(c, u, d, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][c, :, d, b, :, a] += 2*H[l, i, :, :].transpose(1, 0)*delta(k, j)
        idx[2] += [(c, u, d, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][c, :, d, b, :, a] += -2*H[k, i, :, :].transpose(1, 0)*delta(l, j)
        idx[2] += [(c, u, d, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][j, l, :, k, :, a] += 1*H[:, i, :, c]*delta(b, d)
        idx[2] += [(j, l, u, k, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][j, l, :, k, :, a] += -1*H[:, i, :, d]*delta(b, c)
        idx[2] += [(j, l, u, k, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][j, l, :, k, :, b] += -1*H[:, i, :, c]*delta(a, d)
        idx[2] += [(j, l, u, k, v, b) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][j, l, :, k, :, b] += 1*H[:, i, :, d]*delta(a, c)
        idx[2] += [(j, l, u, k, v, b) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][j, :, :, l, :, k] += 1*H[:, i, :, :].transpose(2, 0, 1)*delta(b, c)*delta(a, d)
        idx[2] += [(j, u, v, l, w, k) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[2][j, :, :, l, :, k] += -1*H[:, i, :, :].transpose(2, 0, 1)*delta(a, c)*delta(b, d)
        idx[2] += [(j, u, v, l, w, k) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[2][c, :, d, b, j, a] += -2*H[k, :, l, i]
        idx[2] += [(c, u, d, b, j, a) for u in range(n)]
        hs[2][c, :, d, b, j, a] += 2*H[l, :, k, i]
        idx[2] += [(c, u, d, b, j, a) for u in range(n)]
        hs[2][c, :, d, b, :, a] += 2*H[:, :, l, i].transpose(1, 0)*delta(k, j)
        idx[2] += [(c, u, d, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][c, :, d, b, :, a] += -2*H[:, :, k, i].transpose(1, 0)*delta(l, j)
        idx[2] += [(c, u, d, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][c, :, d, b, :, a] += -2*H[l, :, :, i]*delta(k, j)
        idx[2] += [(c, u, d, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][c, :, d, b, :, a] += 2*H[k, :, :, i]*delta(l, j)
        idx[2] += [(c, u, d, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][j, l, :, k, :, a] += -1*H[:, c, :, i]*delta(b, d)
        idx[2] += [(j, l, u, k, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][j, l, :, k, :, a] += 1*H[:, d, :, i]*delta(b, c)
        idx[2] += [(j, l, u, k, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][j, l, :, k, :, b] += 1*H[:, c, :, i]*delta(a, d)
        idx[2] += [(j, l, u, k, v, b) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][j, l, :, k, :, b] += -1*H[:, d, :, i]*delta(a, c)
        idx[2] += [(j, l, u, k, v, b) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][j, :, :, l, :, k] += -1*H[:, :, :, i].transpose(1, 0, 2)*delta(b, c)*delta(a, d)
        idx[2] += [(j, u, v, l, w, k) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[2][j, :, :, l, :, k] += 1*H[:, :, :, i].transpose(1, 0, 2)*delta(a, c)*delta(b, d)
        idx[2] += [(j, u, v, l, w, k) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[1][c, :, d, :] += 1*H[a, :, b, :].transpose(1, 0)*delta(k, i)*delta(l, j)
        idx[1] += [(c, u, d, v) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[1][c, :, d, :] += -1*H[a, :, b, :].transpose(1, 0)*delta(l, i)*delta(k, j)
        idx[1] += [(c, u, d, v) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][c, :, d, :, i, l] += 1*H[a, :, b, :].transpose(1, 0)*delta(k, j)
        idx[2] += [(c, u, d, v, i, l) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][c, :, d, :, i, k] += -1*H[a, :, b, :].transpose(1, 0)*delta(l, j)
        idx[2] += [(c, u, d, v, i, k) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][c, :, d, :, j, l] += -1*H[a, :, b, :].transpose(1, 0)*delta(k, i)
        idx[2] += [(c, u, d, v, j, l) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][c, :, d, :, j, k] += 1*H[a, :, b, :].transpose(1, 0)*delta(l, i)
        idx[2] += [(c, u, d, v, j, k) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[1][i, l, j, k] += 2*H[a, d, b, c]
        idx[1] += [(i, l, j, k)]
        hs[1][i, l, j, k] += -2*H[a, c, b, d]
        idx[1] += [(i, l, j, k)]
        hs[2][c, :, i, l, j, k] += 2*H[a, :, b, d]
        idx[2] += [(c, u, i, l, j, k) for u in range(n)]
        hs[2][c, :, i, l, j, k] += -2*H[a, d, b, :]
        idx[2] += [(c, u, i, l, j, k) for u in range(n)]
        hs[2][d, :, i, l, j, k] += -2*H[a, :, b, c]
        idx[2] += [(d, u, i, l, j, k) for u in range(n)]
        hs[2][d, :, i, l, j, k] += 2*H[a, c, b, :]
        idx[2] += [(d, u, i, l, j, k) for u in range(n)]
        hs[1][c, :, d, :] += -1*H[b, :, a, :].transpose(1, 0)*delta(k, i)*delta(l, j)
        idx[1] += [(c, u, d, v) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[1][c, :, d, :] += 1*H[b, :, a, :].transpose(1, 0)*delta(l, i)*delta(k, j)
        idx[1] += [(c, u, d, v) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][c, :, d, :, i, l] += -1*H[b, :, a, :].transpose(1, 0)*delta(k, j)
        idx[2] += [(c, u, d, v, i, l) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][c, :, d, :, i, k] += 1*H[b, :, a, :].transpose(1, 0)*delta(l, j)
        idx[2] += [(c, u, d, v, i, k) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][c, :, d, :, j, l] += 1*H[b, :, a, :].transpose(1, 0)*delta(k, i)
        idx[2] += [(c, u, d, v, j, l) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][c, :, d, :, j, k] += -1*H[b, :, a, :].transpose(1, 0)*delta(l, i)
        idx[2] += [(c, u, d, v, j, k) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[1][i, l, j, k] += -2*H[b, d, a, c]
        idx[1] += [(i, l, j, k)]
        hs[1][i, l, j, k] += 2*H[b, c, a, d]
        idx[1] += [(i, l, j, k)]
        hs[2][c, :, i, l, j, k] += -2*H[b, :, a, d]
        idx[2] += [(c, u, i, l, j, k) for u in range(n)]
        hs[2][c, :, i, l, j, k] += 2*H[b, d, a, :]
        idx[2] += [(c, u, i, l, j, k) for u in range(n)]
        hs[2][d, :, i, l, j, k] += 2*H[b, :, a, c]
        idx[2] += [(d, u, i, l, j, k) for u in range(n)]
        hs[2][d, :, i, l, j, k] += -2*H[b, c, a, :]
        idx[2] += [(d, u, i, l, j, k) for u in range(n)]
        hs[2][c, :, d, :, i, b] += 1*H[l, :, a, :].transpose(1, 0)*delta(k, j)
        idx[2] += [(c, u, d, v, i, b) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][c, :, d, :, i, b] += -1*H[k, :, a, :].transpose(1, 0)*delta(l, j)
        idx[2] += [(c, u, d, v, i, b) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][c, :, d, :, j, b] += -1*H[l, :, a, :].transpose(1, 0)*delta(k, i)
        idx[2] += [(c, u, d, v, j, b) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][c, :, d, :, j, b] += 1*H[k, :, a, :].transpose(1, 0)*delta(l, i)
        idx[2] += [(c, u, d, v, j, b) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][c, :, d, :, :, b] += 1*H[:, :, a, :].transpose(2, 1, 0)*delta(k, i)*delta(l, j)
        idx[2] += [(c, u, d, v, w, b) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[2][c, :, d, :, :, b] += -1*H[:, :, a, :].transpose(2, 1, 0)*delta(l, i)*delta(k, j)
        idx[2] += [(c, u, d, v, w, b) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[2][i, l, j, k, :, b] += 2*H[:, d, a, c]
        idx[2] += [(i, l, j, k, u, b) for u in range(n)]
        hs[2][i, l, j, k, :, b] += -2*H[:, c, a, d]
        idx[2] += [(i, l, j, k, u, b) for u in range(n)]
        hs[2][i, :, j, l, :, k] += -2*H[:, :, a, c].transpose(1, 0)*delta(b, d)
        idx[2] += [(i, u, j, l, v, k) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][i, :, j, l, :, k] += 2*H[:, :, a, d].transpose(1, 0)*delta(b, c)
        idx[2] += [(i, u, j, l, v, k) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][i, :, j, l, :, k] += 2*H[:, c, a, :].transpose(1, 0)*delta(b, d)
        idx[2] += [(i, u, j, l, v, k) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][i, :, j, l, :, k] += -2*H[:, d, a, :].transpose(1, 0)*delta(b, c)
        idx[2] += [(i, u, j, l, v, k) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][c, :, d, :, i, a] += -1*H[l, :, b, :].transpose(1, 0)*delta(k, j)
        idx[2] += [(c, u, d, v, i, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][c, :, d, :, i, a] += 1*H[k, :, b, :].transpose(1, 0)*delta(l, j)
        idx[2] += [(c, u, d, v, i, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][c, :, d, :, j, a] += 1*H[l, :, b, :].transpose(1, 0)*delta(k, i)
        idx[2] += [(c, u, d, v, j, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][c, :, d, :, j, a] += -1*H[k, :, b, :].transpose(1, 0)*delta(l, i)
        idx[2] += [(c, u, d, v, j, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][c, :, d, :, :, a] += -1*H[:, :, b, :].transpose(2, 1, 0)*delta(k, i)*delta(l, j)
        idx[2] += [(c, u, d, v, w, a) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[2][c, :, d, :, :, a] += 1*H[:, :, b, :].transpose(2, 1, 0)*delta(l, i)*delta(k, j)
        idx[2] += [(c, u, d, v, w, a) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[2][i, l, j, k, :, a] += -2*H[:, d, b, c]
        idx[2] += [(i, l, j, k, u, a) for u in range(n)]
        hs[2][i, l, j, k, :, a] += 2*H[:, c, b, d]
        idx[2] += [(i, l, j, k, u, a) for u in range(n)]
        hs[2][i, :, j, l, :, k] += 2*H[:, :, b, c].transpose(1, 0)*delta(a, d)
        idx[2] += [(i, u, j, l, v, k) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][i, :, j, l, :, k] += -2*H[:, :, b, d].transpose(1, 0)*delta(a, c)
        idx[2] += [(i, u, j, l, v, k) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][i, :, j, l, :, k] += -2*H[:, c, b, :].transpose(1, 0)*delta(a, d)
        idx[2] += [(i, u, j, l, v, k) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][i, :, j, l, :, k] += 2*H[:, d, b, :].transpose(1, 0)*delta(a, c)
        idx[2] += [(i, u, j, l, v, k) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][c, :, d, :, i, b] += -1*H[a, :, l, :].transpose(1, 0)*delta(k, j)
        idx[2] += [(c, u, d, v, i, b) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][c, :, d, :, i, b] += 1*H[a, :, k, :].transpose(1, 0)*delta(l, j)
        idx[2] += [(c, u, d, v, i, b) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][c, :, d, :, j, b] += 1*H[a, :, l, :].transpose(1, 0)*delta(k, i)
        idx[2] += [(c, u, d, v, j, b) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][c, :, d, :, j, b] += -1*H[a, :, k, :].transpose(1, 0)*delta(l, i)
        idx[2] += [(c, u, d, v, j, b) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][c, :, d, :, :, b] += -1*H[a, :, :, :].transpose(2, 0, 1)*delta(k, i)*delta(l, j)
        idx[2] += [(c, u, d, v, w, b) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[2][c, :, d, :, :, b] += 1*H[a, :, :, :].transpose(2, 0, 1)*delta(l, i)*delta(k, j)
        idx[2] += [(c, u, d, v, w, b) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[2][i, l, j, k, :, b] += -2*H[a, d, :, c]
        idx[2] += [(i, l, j, k, u, b) for u in range(n)]
        hs[2][i, l, j, k, :, b] += 2*H[a, c, :, d]
        idx[2] += [(i, l, j, k, u, b) for u in range(n)]
        hs[2][i, :, j, l, :, k] += 2*H[a, :, :, c]*delta(b, d)
        idx[2] += [(i, u, j, l, v, k) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][i, :, j, l, :, k] += -2*H[a, :, :, d]*delta(b, c)
        idx[2] += [(i, u, j, l, v, k) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][i, :, j, l, :, k] += -2*H[a, c, :, :].transpose(1, 0)*delta(b, d)
        idx[2] += [(i, u, j, l, v, k) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][i, :, j, l, :, k] += 2*H[a, d, :, :].transpose(1, 0)*delta(b, c)
        idx[2] += [(i, u, j, l, v, k) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][c, :, d, :, i, a] += 1*H[b, :, l, :].transpose(1, 0)*delta(k, j)
        idx[2] += [(c, u, d, v, i, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][c, :, d, :, i, a] += -1*H[b, :, k, :].transpose(1, 0)*delta(l, j)
        idx[2] += [(c, u, d, v, i, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][c, :, d, :, j, a] += -1*H[b, :, l, :].transpose(1, 0)*delta(k, i)
        idx[2] += [(c, u, d, v, j, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][c, :, d, :, j, a] += 1*H[b, :, k, :].transpose(1, 0)*delta(l, i)
        idx[2] += [(c, u, d, v, j, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][c, :, d, :, :, a] += 1*H[b, :, :, :].transpose(2, 0, 1)*delta(k, i)*delta(l, j)
        idx[2] += [(c, u, d, v, w, a) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[2][c, :, d, :, :, a] += -1*H[b, :, :, :].transpose(2, 0, 1)*delta(l, i)*delta(k, j)
        idx[2] += [(c, u, d, v, w, a) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[2][i, l, j, k, :, a] += 2*H[b, d, :, c]
        idx[2] += [(i, l, j, k, u, a) for u in range(n)]
        hs[2][i, l, j, k, :, a] += -2*H[b, c, :, d]
        idx[2] += [(i, l, j, k, u, a) for u in range(n)]
        hs[2][i, :, j, l, :, k] += -2*H[b, :, :, c]*delta(a, d)
        idx[2] += [(i, u, j, l, v, k) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][i, :, j, l, :, k] += 2*H[b, :, :, d]*delta(a, c)
        idx[2] += [(i, u, j, l, v, k) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][i, :, j, l, :, k] += 2*H[b, c, :, :].transpose(1, 0)*delta(a, d)
        idx[2] += [(i, u, j, l, v, k) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][i, :, j, l, :, k] += -2*H[b, d, :, :].transpose(1, 0)*delta(a, c)
        idx[2] += [(i, u, j, l, v, k) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[3][d, :, j, l, :, k, :, b] += 1*H[:, :, :, i].transpose(1, 0, 2)*delta(a, c)
        idx[3] += [(d, u, j, l, v, k, w, b) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[3][d, :, j, l, :, k, :, a] += -1*H[:, :, :, i].transpose(1, 0, 2)*delta(b, c)
        idx[3] += [(d, u, j, l, v, k, w, a) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[3][d, l, j, k, :, b, :, a] += 2*H[:, c, :, i]
        idx[3] += [(d, l, j, k, u, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[3][c, :, j, l, :, k, :, b] += -1*H[:, :, :, i].transpose(1, 0, 2)*delta(a, d)
        idx[3] += [(c, u, j, l, v, k, w, b) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[3][c, :, j, l, :, k, :, a] += 1*H[:, :, :, i].transpose(1, 0, 2)*delta(b, d)
        idx[3] += [(c, u, j, l, v, k, w, a) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[3][c, l, j, k, :, b, :, a] += -2*H[:, d, :, i]
        idx[3] += [(c, l, j, k, u, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[3][d, :, i, :, j, l, :, k] += -1*H[:, :, a, :].transpose(2, 1, 0)*delta(b, c)
        idx[3] += [(d, u, i, v, j, l, w, k) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[3][d, :, i, l, j, k, :, b] += 2*H[:, c, a, :].transpose(1, 0)
        idx[3] += [(d, u, i, l, j, k, v, b) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[3][d, :, i, l, j, k, :, b] += -2*H[:, :, a, c].transpose(1, 0)
        idx[3] += [(d, u, i, l, j, k, v, b) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[3][c, :, i, :, j, l, :, k] += 1*H[:, :, a, :].transpose(2, 1, 0)*delta(b, d)
        idx[3] += [(c, u, i, v, j, l, w, k) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[3][c, :, i, l, j, k, :, b] += -2*H[:, d, a, :].transpose(1, 0)
        idx[3] += [(c, u, i, l, j, k, v, b) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[3][c, :, i, l, j, k, :, b] += 2*H[:, :, a, d].transpose(1, 0)
        idx[3] += [(c, u, i, l, j, k, v, b) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[3][c, :, d, k, :, b, :, a] += 1*H[:, :, :, i].transpose(1, 0, 2)*delta(l, j)
        idx[3] += [(c, u, d, k, v, b, w, a) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[3][c, :, d, l, :, b, :, a] += -1*H[:, :, :, i].transpose(1, 0, 2)*delta(k, j)
        idx[3] += [(c, u, d, l, v, b, w, a) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[3][c, :, d, k, j, b, :, a] += -2*H[l, :, :, i]
        idx[3] += [(c, u, d, k, j, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[3][c, :, d, l, j, b, :, a] += 2*H[k, :, :, i]
        idx[3] += [(c, u, d, l, j, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[3][c, :, d, k, j, b, :, a] += 2*H[:, :, l, i].transpose(1, 0)
        idx[3] += [(c, u, d, k, j, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[3][c, :, d, l, j, b, :, a] += -2*H[:, :, k, i].transpose(1, 0)
        idx[3] += [(c, u, d, l, j, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[3][c, :, d, :, j, k, :, b] += -1*H[a, :, :, :].transpose(2, 0, 1)*delta(l, i)
        idx[3] += [(c, u, d, v, j, k, w, b) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[3][c, :, d, :, j, l, :, b] += 1*H[a, :, :, :].transpose(2, 0, 1)*delta(k, i)
        idx[3] += [(c, u, d, v, j, l, w, b) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[3][c, :, d, :, i, k, :, b] += 1*H[a, :, :, :].transpose(2, 0, 1)*delta(l, j)
        idx[3] += [(c, u, d, v, i, k, w, b) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[3][c, :, d, :, i, l, :, b] += -1*H[a, :, :, :].transpose(2, 0, 1)*delta(k, j)
        idx[3] += [(c, u, d, v, i, l, w, b) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[3][c, :, d, :, i, k, j, b] += -2*H[a, :, l, :].transpose(1, 0)
        idx[3] += [(c, u, d, v, i, k, j, b) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[3][c, :, d, :, i, l, j, b] += 2*H[a, :, k, :].transpose(1, 0)
        idx[3] += [(c, u, d, v, i, l, j, b) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[3][i, :, j, l, :, k, :, a] += 1*H[:, :, :, c].transpose(1, 0, 2)*delta(b, d)
        idx[3] += [(i, u, j, l, v, k, w, a) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[3][i, :, j, l, :, k, :, b] += -1*H[:, :, :, c].transpose(1, 0, 2)*delta(a, d)
        idx[3] += [(i, u, j, l, v, k, w, b) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[3][d, :, j, l, :, b, :, a] += -1*H[:, :, :, c].transpose(1, 0, 2)*delta(k, i)
        idx[3] += [(d, u, j, l, v, b, w, a) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[3][d, :, j, k, :, b, :, a] += 1*H[:, :, :, c].transpose(1, 0, 2)*delta(l, i)
        idx[3] += [(d, u, j, k, v, b, w, a) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[3][d, :, i, l, :, b, :, a] += 1*H[:, :, :, c].transpose(1, 0, 2)*delta(k, j)
        idx[3] += [(d, u, i, l, v, b, w, a) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[3][d, :, i, k, :, b, :, a] += -1*H[:, :, :, c].transpose(1, 0, 2)*delta(l, j)
        idx[3] += [(d, u, i, k, v, b, w, a) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[3][d, :, j, l, :, k, :, b] += -1*H[:, i, :, :].transpose(2, 0, 1)*delta(a, c)
        idx[3] += [(d, u, j, l, v, k, w, b) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[3][d, :, j, l, :, k, :, a] += 1*H[:, i, :, :].transpose(2, 0, 1)*delta(b, c)
        idx[3] += [(d, u, j, l, v, k, w, a) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[3][d, l, j, k, :, b, :, a] += -2*H[:, i, :, c]
        idx[3] += [(d, l, j, k, u, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[3][c, :, j, l, :, k, :, b] += 1*H[:, i, :, :].transpose(2, 0, 1)*delta(a, d)
        idx[3] += [(c, u, j, l, v, k, w, b) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[3][c, :, j, l, :, k, :, a] += -1*H[:, i, :, :].transpose(2, 0, 1)*delta(b, d)
        idx[3] += [(c, u, j, l, v, k, w, a) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[3][c, l, j, k, :, b, :, a] += 2*H[:, i, :, d]
        idx[3] += [(c, l, j, k, u, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[3][i, :, j, l, :, k, :, a] += -1*H[:, c, :, :].transpose(2, 0, 1)*delta(b, d)
        idx[3] += [(i, u, j, l, v, k, w, a) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[3][i, :, j, l, :, k, :, b] += 1*H[:, c, :, :].transpose(2, 0, 1)*delta(a, d)
        idx[3] += [(i, u, j, l, v, k, w, b) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[3][d, :, j, l, :, b, :, a] += 1*H[:, c, :, :].transpose(2, 0, 1)*delta(k, i)
        idx[3] += [(d, u, j, l, v, b, w, a) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[3][d, :, j, k, :, b, :, a] += -1*H[:, c, :, :].transpose(2, 0, 1)*delta(l, i)
        idx[3] += [(d, u, j, k, v, b, w, a) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[3][d, :, i, l, :, b, :, a] += -1*H[:, c, :, :].transpose(2, 0, 1)*delta(k, j)
        idx[3] += [(d, u, i, l, v, b, w, a) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[3][d, :, i, k, :, b, :, a] += 1*H[:, c, :, :].transpose(2, 0, 1)*delta(l, j)
        idx[3] += [(d, u, i, k, v, b, w, a) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[3][c, :, d, :, j, k, :, b] += 1*H[:, :, a, :].transpose(2, 1, 0)*delta(l, i)
        idx[3] += [(c, u, d, v, j, k, w, b) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[3][c, :, d, :, j, l, :, b] += -1*H[:, :, a, :].transpose(2, 1, 0)*delta(k, i)
        idx[3] += [(c, u, d, v, j, l, w, b) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[3][c, :, d, :, i, k, :, b] += -1*H[:, :, a, :].transpose(2, 1, 0)*delta(l, j)
        idx[3] += [(c, u, d, v, i, k, w, b) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[3][c, :, d, :, i, l, :, b] += 1*H[:, :, a, :].transpose(2, 1, 0)*delta(k, j)
        idx[3] += [(c, u, d, v, i, l, w, b) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[3][c, :, d, :, i, k, j, b] += 2*H[l, :, a, :].transpose(1, 0)
        idx[3] += [(c, u, d, v, i, k, j, b) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[3][c, :, d, :, i, l, j, b] += -2*H[k, :, a, :].transpose(1, 0)
        idx[3] += [(c, u, d, v, i, l, j, b) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[3][i, :, j, l, :, k, :, a] += -1*H[:, :, :, d].transpose(1, 0, 2)*delta(b, c)
        idx[3] += [(i, u, j, l, v, k, w, a) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[3][i, :, j, l, :, k, :, b] += 1*H[:, :, :, d].transpose(1, 0, 2)*delta(a, c)
        idx[3] += [(i, u, j, l, v, k, w, b) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[3][c, :, j, l, :, b, :, a] += 1*H[:, :, :, d].transpose(1, 0, 2)*delta(k, i)
        idx[3] += [(c, u, j, l, v, b, w, a) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[3][c, :, j, k, :, b, :, a] += -1*H[:, :, :, d].transpose(1, 0, 2)*delta(l, i)
        idx[3] += [(c, u, j, k, v, b, w, a) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[3][c, :, i, l, :, b, :, a] += -1*H[:, :, :, d].transpose(1, 0, 2)*delta(k, j)
        idx[3] += [(c, u, i, l, v, b, w, a) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[3][c, :, i, k, :, b, :, a] += 1*H[:, :, :, d].transpose(1, 0, 2)*delta(l, j)
        idx[3] += [(c, u, i, k, v, b, w, a) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[3][c, :, d, k, :, b, :, a] += -1*H[:, i, :, :].transpose(2, 0, 1)*delta(l, j)
        idx[3] += [(c, u, d, k, v, b, w, a) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[3][c, :, d, l, :, b, :, a] += 1*H[:, i, :, :].transpose(2, 0, 1)*delta(k, j)
        idx[3] += [(c, u, d, l, v, b, w, a) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[3][c, :, d, k, j, b, :, a] += 2*H[l, i, :, :].transpose(1, 0)
        idx[3] += [(c, u, d, k, j, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[3][c, :, d, l, j, b, :, a] += -2*H[k, i, :, :].transpose(1, 0)
        idx[3] += [(c, u, d, l, j, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[3][c, :, d, k, j, b, :, a] += -2*H[:, i, l, :].transpose(1, 0)
        idx[3] += [(c, u, d, k, j, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[3][c, :, d, l, j, b, :, a] += 2*H[:, i, k, :].transpose(1, 0)
        idx[3] += [(c, u, d, l, j, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[3][i, :, j, l, :, k, :, a] += 1*H[:, d, :, :].transpose(2, 0, 1)*delta(b, c)
        idx[3] += [(i, u, j, l, v, k, w, a) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[3][i, :, j, l, :, k, :, b] += -1*H[:, d, :, :].transpose(2, 0, 1)*delta(a, c)
        idx[3] += [(i, u, j, l, v, k, w, b) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[3][c, :, j, l, :, b, :, a] += -1*H[:, d, :, :].transpose(2, 0, 1)*delta(k, i)
        idx[3] += [(c, u, j, l, v, b, w, a) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[3][c, :, j, k, :, b, :, a] += 1*H[:, d, :, :].transpose(2, 0, 1)*delta(l, i)
        idx[3] += [(c, u, j, k, v, b, w, a) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[3][c, :, i, l, :, b, :, a] += 1*H[:, d, :, :].transpose(2, 0, 1)*delta(k, j)
        idx[3] += [(c, u, i, l, v, b, w, a) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[3][c, :, i, k, :, b, :, a] += -1*H[:, d, :, :].transpose(2, 0, 1)*delta(l, j)
        idx[3] += [(c, u, i, k, v, b, w, a) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[3][d, :, i, :, j, l, :, k] += -1*H[b, :, :, :].transpose(2, 0, 1)*delta(a, c)
        idx[3] += [(d, u, i, v, j, l, w, k) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[3][d, :, i, l, j, k, :, a] += 2*H[b, c, :, :].transpose(1, 0)
        idx[3] += [(d, u, i, l, j, k, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[3][d, :, i, l, j, k, :, a] += -2*H[b, :, :, c]
        idx[3] += [(d, u, i, l, j, k, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[3][c, :, i, :, j, l, :, k] += 1*H[b, :, :, :].transpose(2, 0, 1)*delta(a, d)
        idx[3] += [(c, u, i, v, j, l, w, k) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[3][c, :, i, l, j, k, :, a] += -2*H[b, d, :, :].transpose(1, 0)
        idx[3] += [(c, u, i, l, j, k, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[3][c, :, i, l, j, k, :, a] += 2*H[b, :, :, d]
        idx[3] += [(c, u, i, l, j, k, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[3][d, :, i, l, :, k, :, b] += -1*H[:, :, :, j].transpose(1, 0, 2)*delta(a, c)
        idx[3] += [(d, u, i, l, v, k, w, b) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[3][d, :, i, l, :, k, :, a] += 1*H[:, :, :, j].transpose(1, 0, 2)*delta(b, c)
        idx[3] += [(d, u, i, l, v, k, w, a) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[3][d, l, i, k, :, b, :, a] += -2*H[:, c, :, j]
        idx[3] += [(d, l, i, k, u, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[3][c, :, i, l, :, k, :, b] += 1*H[:, :, :, j].transpose(1, 0, 2)*delta(a, d)
        idx[3] += [(c, u, i, l, v, k, w, b) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[3][c, :, i, l, :, k, :, a] += -1*H[:, :, :, j].transpose(1, 0, 2)*delta(b, d)
        idx[3] += [(c, u, i, l, v, k, w, a) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[3][c, l, i, k, :, b, :, a] += 2*H[:, d, :, j]
        idx[3] += [(c, l, i, k, u, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[3][d, :, i, :, j, l, :, k] += 1*H[:, :, b, :].transpose(2, 1, 0)*delta(a, c)
        idx[3] += [(d, u, i, v, j, l, w, k) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[3][d, :, i, l, j, k, :, a] += -2*H[:, c, b, :].transpose(1, 0)
        idx[3] += [(d, u, i, l, j, k, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[3][d, :, i, l, j, k, :, a] += 2*H[:, :, b, c].transpose(1, 0)
        idx[3] += [(d, u, i, l, j, k, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[3][c, :, i, :, j, l, :, k] += -1*H[:, :, b, :].transpose(2, 1, 0)*delta(a, d)
        idx[3] += [(c, u, i, v, j, l, w, k) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[3][c, :, i, l, j, k, :, a] += 2*H[:, d, b, :].transpose(1, 0)
        idx[3] += [(c, u, i, l, j, k, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[3][c, :, i, l, j, k, :, a] += -2*H[:, :, b, d].transpose(1, 0)
        idx[3] += [(c, u, i, l, j, k, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[3][d, :, i, :, j, k, :, a] += -1*H[l, :, :, :].transpose(2, 0, 1)*delta(b, c)
        idx[3] += [(d, u, i, v, j, k, w, a) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[3][d, :, i, :, j, k, :, b] += 1*H[l, :, :, :].transpose(2, 0, 1)*delta(a, c)
        idx[3] += [(d, u, i, v, j, k, w, b) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[3][c, :, i, :, j, k, :, a] += 1*H[l, :, :, :].transpose(2, 0, 1)*delta(b, d)
        idx[3] += [(c, u, i, v, j, k, w, a) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[3][c, :, i, :, j, k, :, b] += -1*H[l, :, :, :].transpose(2, 0, 1)*delta(a, d)
        idx[3] += [(c, u, i, v, j, k, w, b) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[3][c, :, d, k, :, b, :, a] += -1*H[:, :, :, j].transpose(1, 0, 2)*delta(l, i)
        idx[3] += [(c, u, d, k, v, b, w, a) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[3][c, :, d, l, :, b, :, a] += 1*H[:, :, :, j].transpose(1, 0, 2)*delta(k, i)
        idx[3] += [(c, u, d, l, v, b, w, a) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[3][c, :, d, k, i, b, :, a] += 2*H[l, :, :, j]
        idx[3] += [(c, u, d, k, i, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[3][c, :, d, l, i, b, :, a] += -2*H[k, :, :, j]
        idx[3] += [(c, u, d, l, i, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[3][c, :, d, k, i, b, :, a] += -2*H[:, :, l, j].transpose(1, 0)
        idx[3] += [(c, u, d, k, i, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[3][c, :, d, l, i, b, :, a] += 2*H[:, :, k, j].transpose(1, 0)
        idx[3] += [(c, u, d, l, i, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[3][c, :, d, :, j, b, :, a] += -1*H[l, :, :, :].transpose(2, 0, 1)*delta(k, i)
        idx[3] += [(c, u, d, v, j, b, w, a) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[3][c, :, d, :, i, b, :, a] += 1*H[l, :, :, :].transpose(2, 0, 1)*delta(k, j)
        idx[3] += [(c, u, d, v, i, b, w, a) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[3][d, :, i, :, j, l, :, a] += 1*H[k, :, :, :].transpose(2, 0, 1)*delta(b, c)
        idx[3] += [(d, u, i, v, j, l, w, a) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[3][d, :, i, :, j, l, :, b] += -1*H[k, :, :, :].transpose(2, 0, 1)*delta(a, c)
        idx[3] += [(d, u, i, v, j, l, w, b) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[3][c, :, i, :, j, l, :, a] += -1*H[k, :, :, :].transpose(2, 0, 1)*delta(b, d)
        idx[3] += [(c, u, i, v, j, l, w, a) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[3][c, :, i, :, j, l, :, b] += 1*H[k, :, :, :].transpose(2, 0, 1)*delta(a, d)
        idx[3] += [(c, u, i, v, j, l, w, b) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[3][d, :, i, :, j, l, :, k] += 1*H[a, :, :, :].transpose(2, 0, 1)*delta(b, c)
        idx[3] += [(d, u, i, v, j, l, w, k) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[3][d, :, i, l, j, k, :, b] += -2*H[a, c, :, :].transpose(1, 0)
        idx[3] += [(d, u, i, l, j, k, v, b) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[3][d, :, i, l, j, k, :, b] += 2*H[a, :, :, c]
        idx[3] += [(d, u, i, l, j, k, v, b) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[3][c, :, i, :, j, l, :, k] += -1*H[a, :, :, :].transpose(2, 0, 1)*delta(b, d)
        idx[3] += [(c, u, i, v, j, l, w, k) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[3][c, :, i, l, j, k, :, b] += 2*H[a, d, :, :].transpose(1, 0)
        idx[3] += [(c, u, i, l, j, k, v, b) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[3][c, :, i, l, j, k, :, b] += -2*H[a, :, :, d]
        idx[3] += [(c, u, i, l, j, k, v, b) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[3][c, :, d, :, j, b, :, a] += 1*H[k, :, :, :].transpose(2, 0, 1)*delta(l, i)
        idx[3] += [(c, u, d, v, j, b, w, a) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[3][c, :, d, :, i, b, :, a] += -1*H[k, :, :, :].transpose(2, 0, 1)*delta(l, j)
        idx[3] += [(c, u, d, v, i, b, w, a) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[3][d, :, i, :, j, k, :, a] += 1*H[:, :, l, :].transpose(2, 1, 0)*delta(b, c)
        idx[3] += [(d, u, i, v, j, k, w, a) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[3][d, :, i, :, j, k, :, b] += -1*H[:, :, l, :].transpose(2, 1, 0)*delta(a, c)
        idx[3] += [(d, u, i, v, j, k, w, b) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[3][c, :, i, :, j, k, :, a] += -1*H[:, :, l, :].transpose(2, 1, 0)*delta(b, d)
        idx[3] += [(c, u, i, v, j, k, w, a) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[3][c, :, i, :, j, k, :, b] += 1*H[:, :, l, :].transpose(2, 1, 0)*delta(a, d)
        idx[3] += [(c, u, i, v, j, k, w, b) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[3][d, :, i, l, :, k, :, b] += 1*H[:, j, :, :].transpose(2, 0, 1)*delta(a, c)
        idx[3] += [(d, u, i, l, v, k, w, b) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[3][d, :, i, l, :, k, :, a] += -1*H[:, j, :, :].transpose(2, 0, 1)*delta(b, c)
        idx[3] += [(d, u, i, l, v, k, w, a) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[3][d, l, i, k, :, b, :, a] += 2*H[:, j, :, c]
        idx[3] += [(d, l, i, k, u, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[3][c, :, i, l, :, k, :, b] += -1*H[:, j, :, :].transpose(2, 0, 1)*delta(a, d)
        idx[3] += [(c, u, i, l, v, k, w, b) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[3][c, :, i, l, :, k, :, a] += 1*H[:, j, :, :].transpose(2, 0, 1)*delta(b, d)
        idx[3] += [(c, u, i, l, v, k, w, a) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[3][c, l, i, k, :, b, :, a] += -2*H[:, j, :, d]
        idx[3] += [(c, l, i, k, u, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[3][c, :, d, :, j, b, :, a] += 1*H[:, :, l, :].transpose(2, 1, 0)*delta(k, i)
        idx[3] += [(c, u, d, v, j, b, w, a) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[3][c, :, d, :, i, b, :, a] += -1*H[:, :, l, :].transpose(2, 1, 0)*delta(k, j)
        idx[3] += [(c, u, d, v, i, b, w, a) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[3][d, :, i, :, j, l, :, a] += -1*H[:, :, k, :].transpose(2, 1, 0)*delta(b, c)
        idx[3] += [(d, u, i, v, j, l, w, a) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[3][d, :, i, :, j, l, :, b] += 1*H[:, :, k, :].transpose(2, 1, 0)*delta(a, c)
        idx[3] += [(d, u, i, v, j, l, w, b) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[3][c, :, i, :, j, l, :, a] += 1*H[:, :, k, :].transpose(2, 1, 0)*delta(b, d)
        idx[3] += [(c, u, i, v, j, l, w, a) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[3][c, :, i, :, j, l, :, b] += -1*H[:, :, k, :].transpose(2, 1, 0)*delta(a, d)
        idx[3] += [(c, u, i, v, j, l, w, b) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[3][c, :, d, :, j, k, :, a] += -1*H[:, :, b, :].transpose(2, 1, 0)*delta(l, i)
        idx[3] += [(c, u, d, v, j, k, w, a) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[3][c, :, d, :, j, l, :, a] += 1*H[:, :, b, :].transpose(2, 1, 0)*delta(k, i)
        idx[3] += [(c, u, d, v, j, l, w, a) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[3][c, :, d, :, i, k, :, a] += 1*H[:, :, b, :].transpose(2, 1, 0)*delta(l, j)
        idx[3] += [(c, u, d, v, i, k, w, a) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[3][c, :, d, :, i, l, :, a] += -1*H[:, :, b, :].transpose(2, 1, 0)*delta(k, j)
        idx[3] += [(c, u, d, v, i, l, w, a) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[3][c, :, d, :, i, k, j, a] += -2*H[l, :, b, :].transpose(1, 0)
        idx[3] += [(c, u, d, v, i, k, j, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[3][c, :, d, :, i, l, j, a] += 2*H[k, :, b, :].transpose(1, 0)
        idx[3] += [(c, u, d, v, i, l, j, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[3][c, :, d, :, j, b, :, a] += -1*H[:, :, k, :].transpose(2, 1, 0)*delta(l, i)
        idx[3] += [(c, u, d, v, j, b, w, a) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[3][c, :, d, :, i, b, :, a] += 1*H[:, :, k, :].transpose(2, 1, 0)*delta(l, j)
        idx[3] += [(c, u, d, v, i, b, w, a) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[3][c, :, d, k, :, b, :, a] += 1*H[:, j, :, :].transpose(2, 0, 1)*delta(l, i)
        idx[3] += [(c, u, d, k, v, b, w, a) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[3][c, :, d, l, :, b, :, a] += -1*H[:, j, :, :].transpose(2, 0, 1)*delta(k, i)
        idx[3] += [(c, u, d, l, v, b, w, a) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[3][c, :, d, k, i, b, :, a] += -2*H[l, j, :, :].transpose(1, 0)
        idx[3] += [(c, u, d, k, i, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[3][c, :, d, l, i, b, :, a] += 2*H[k, j, :, :].transpose(1, 0)
        idx[3] += [(c, u, d, l, i, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[3][c, :, d, k, i, b, :, a] += 2*H[:, j, l, :].transpose(1, 0)
        idx[3] += [(c, u, d, k, i, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[3][c, :, d, l, i, b, :, a] += -2*H[:, j, k, :].transpose(1, 0)
        idx[3] += [(c, u, d, l, i, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[3][c, :, d, :, j, k, :, a] += 1*H[b, :, :, :].transpose(2, 0, 1)*delta(l, i)
        idx[3] += [(c, u, d, v, j, k, w, a) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[3][c, :, d, :, j, l, :, a] += -1*H[b, :, :, :].transpose(2, 0, 1)*delta(k, i)
        idx[3] += [(c, u, d, v, j, l, w, a) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[3][c, :, d, :, i, k, :, a] += -1*H[b, :, :, :].transpose(2, 0, 1)*delta(l, j)
        idx[3] += [(c, u, d, v, i, k, w, a) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[3][c, :, d, :, i, l, :, a] += 1*H[b, :, :, :].transpose(2, 0, 1)*delta(k, j)
        idx[3] += [(c, u, d, v, i, l, w, a) for (u, v, w) in itertools.product(range(n), repeat=3)]
        hs[3][c, :, d, :, i, k, j, a] += 2*H[b, :, l, :].transpose(1, 0)
        idx[3] += [(c, u, d, v, i, k, j, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[3][c, :, d, :, i, l, j, a] += -2*H[b, :, k, :].transpose(1, 0)
        idx[3] += [(c, u, d, v, i, l, j, a) for (u, v) in itertools.product(range(n), repeat=2)]
    for k in range(4):
        if(idx[k] is not None):
            idx[k] = list(set(idx[k]))
    return FermionicOperatorNBody(hs), idx


def triple_commutator_adj_twobody_adj(n, Emu, Enu, H):
    """
      Args:
      n [int]:          number of orbitals
      Emu [list]:       excitation operator
      Enu [list]:       excitation operator
      H [rank-2 float]: matrix corresponding to H[i,j,k,l] \hat{a}^\dagger_i \hat{a}^\dagger_j \hat{a}_k \hat{a}_l
      constructs the FermionicOperator representation of operator
      Q_{IJ} = (Psi|[dag(E_I),H,dag(E_J)]|Psi)
    """
    hs = [None]*4
    idx = [None]*4
    if(len(Emu) == 2 and len(Enu) == 2):
        a, i = Emu
        b, j = Enu
        hs[1] = np.zeros(tuple([n]*4))
        idx[1] = []
        hs[1][i, :, :, b] += -2*H[a, j, :, :].transpose(1, 0)
        idx[1] += [(i, u, v, b) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[1][i, :, :, b] += 2*H[a, :, :, j]
        idx[1] += [(i, u, v, b) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[1][i, :, j, :] += 2*H[a, :, b, :].transpose(1, 0)
        idx[1] += [(i, u, j, v) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[1][i, :, :, b] += 2*H[:, j, a, :].transpose(1, 0)
        idx[1] += [(i, u, v, b) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[1][i, :, :, b] += -2*H[:, :, a, j].transpose(1, 0)
        idx[1] += [(i, u, v, b) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[1][i, :, j, :] += -2*H[b, :, a, :].transpose(1, 0)
        idx[1] += [(i, u, j, v) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[1][:, b, :, a] += -2*H[:, j, :, i]
        idx[1] += [(u, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[1][j, :, :, a] += 2*H[b, :, :, i]
        idx[1] += [(j, u, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[1][j, :, :, a] += -2*H[:, :, b, i].transpose(1, 0)
        idx[1] += [(j, u, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[1][:, b, :, a] += 2*H[:, i, :, j]
        idx[1] += [(u, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[1][j, :, :, a] += -2*H[b, i, :, :].transpose(1, 0)
        idx[1] += [(j, u, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[1][j, :, :, a] += 2*H[:, i, b, :].transpose(1, 0)
        idx[1] += [(j, u, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
    if(len(Emu) == 2 and len(Enu) == 4):
        a, i = Emu
        b, c, k, j = Enu
        hs[1] = np.zeros(tuple([n]*4))
        idx[1] = []
        hs[2] = np.zeros(tuple([n]*6))
        idx[2] = []
        hs[1][j, :, k, a] += -2*H[b, i, c, :]
        idx[1] += [(j, u, k, a) for u in range(n)]
        hs[1][j, :, k, a] += 2*H[c, i, b, :]
        idx[1] += [(j, u, k, a) for u in range(n)]
        hs[1][j, :, k, a] += 2*H[b, :, c, i]
        idx[1] += [(j, u, k, a) for u in range(n)]
        hs[1][j, :, k, a] += -2*H[c, :, b, i]
        idx[1] += [(j, u, k, a) for u in range(n)]
        hs[1][i, c, :, b] += -2*H[:, k, a, j]
        idx[1] += [(i, c, u, b) for u in range(n)]
        hs[1][i, c, :, b] += 2*H[:, j, a, k]
        idx[1] += [(i, c, u, b) for u in range(n)]
        hs[1][i, c, :, b] += 2*H[a, k, :, j]
        idx[1] += [(i, c, u, b) for u in range(n)]
        hs[1][i, c, :, b] += -2*H[a, j, :, k]
        idx[1] += [(i, c, u, b) for u in range(n)]
        hs[2][i, :, k, c, :, b] += 2*H[:, j, a, :].transpose(1, 0)
        idx[2] += [(i, u, k, c, v, b) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][i, :, k, c, :, b] += -2*H[:, :, a, j].transpose(1, 0)
        idx[2] += [(i, u, k, c, v, b) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][i, :, j, c, :, b] += -2*H[:, k, a, :].transpose(1, 0)
        idx[2] += [(i, u, j, c, v, b) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][i, :, j, c, :, b] += 2*H[:, :, a, k].transpose(1, 0)
        idx[2] += [(i, u, j, c, v, b) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][i, :, j, c, :, b] += 2*H[a, k, :, :].transpose(1, 0)
        idx[2] += [(i, u, j, c, v, b) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][i, :, j, c, :, b] += -2*H[a, :, :, k]
        idx[2] += [(i, u, j, c, v, b) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][i, :, j, :, k, b] += 2*H[c, :, a, :].transpose(1, 0)
        idx[2] += [(i, u, j, v, k, b) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][i, :, j, :, k, c] += -2*H[b, :, a, :].transpose(1, 0)
        idx[2] += [(i, u, j, v, k, c) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][k, c, :, b, :, a] += -2*H[:, j, :, i]
        idx[2] += [(k, c, u, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][j, c, :, b, :, a] += 2*H[:, k, :, i]
        idx[2] += [(j, c, u, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][j, :, k, b, :, a] += -2*H[c, :, :, i]
        idx[2] += [(j, u, k, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][j, :, k, c, :, a] += 2*H[b, :, :, i]
        idx[2] += [(j, u, k, c, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][j, :, k, b, :, a] += 2*H[:, :, c, i].transpose(1, 0)
        idx[2] += [(j, u, k, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][j, :, k, c, :, a] += -2*H[:, :, b, i].transpose(1, 0)
        idx[2] += [(j, u, k, c, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][i, :, k, c, :, b] += -2*H[a, j, :, :].transpose(1, 0)
        idx[2] += [(i, u, k, c, v, b) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][i, :, k, c, :, b] += 2*H[a, :, :, j]
        idx[2] += [(i, u, k, c, v, b) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][k, c, :, b, :, a] += 2*H[:, i, :, j]
        idx[2] += [(k, c, u, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][j, c, :, b, :, a] += -2*H[:, i, :, k]
        idx[2] += [(j, c, u, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][j, :, k, b, :, a] += 2*H[c, i, :, :].transpose(1, 0)
        idx[2] += [(j, u, k, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][j, :, k, c, :, a] += -2*H[b, i, :, :].transpose(1, 0)
        idx[2] += [(j, u, k, c, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][j, :, k, b, :, a] += -2*H[:, i, c, :].transpose(1, 0)
        idx[2] += [(j, u, k, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][j, :, k, c, :, a] += 2*H[:, i, b, :].transpose(1, 0)
        idx[2] += [(j, u, k, c, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][i, :, j, :, k, b] += -2*H[a, :, c, :].transpose(1, 0)
        idx[2] += [(i, u, j, v, k, b) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[2][i, :, j, :, k, c] += 2*H[a, :, b, :].transpose(1, 0)
        idx[2] += [(i, u, j, v, k, c) for (u, v) in itertools.product(range(n), repeat=2)]
    if(len(Emu) == 4 and len(Enu) == 4):
        a, b, j, i = Emu
        c, d, l, k = Enu
        hs[1] = np.zeros(tuple([n]*4))
        idx[1] = []
        hs[2] = np.zeros(tuple([n]*6))
        idx[2] = []
        hs[3] = np.zeros(tuple([n]*8))
        idx[3] = []
        hs[1][k, b, l, a] += -2*H[c, i, d, j]
        idx[1] += [(k, b, l, a)]
        hs[1][k, b, l, a] += 2*H[d, i, c, j]
        idx[1] += [(k, b, l, a)]
        hs[2][k, d, l, b, :, a] += -2*H[:, i, c, j]
        idx[2] += [(k, d, l, b, u, a) for u in range(n)]
        hs[2][k, c, l, b, :, a] += 2*H[:, i, d, j]
        idx[2] += [(k, c, l, b, u, a) for u in range(n)]
        hs[2][k, d, l, b, :, a] += 2*H[c, i, :, j]
        idx[2] += [(k, d, l, b, u, a) for u in range(n)]
        hs[2][k, c, l, b, :, a] += -2*H[d, i, :, j]
        idx[2] += [(k, c, l, b, u, a) for u in range(n)]
        hs[1][k, b, l, a] += 2*H[c, j, d, i]
        idx[1] += [(k, b, l, a)]
        hs[1][k, b, l, a] += -2*H[d, j, c, i]
        idx[1] += [(k, b, l, a)]
        hs[2][k, d, l, b, :, a] += 2*H[:, j, c, i]
        idx[2] += [(k, d, l, b, u, a) for u in range(n)]
        hs[2][k, c, l, b, :, a] += -2*H[:, j, d, i]
        idx[2] += [(k, c, l, b, u, a) for u in range(n)]
        hs[2][k, d, l, b, :, a] += -2*H[c, j, :, i]
        idx[2] += [(k, d, l, b, u, a) for u in range(n)]
        hs[2][k, c, l, b, :, a] += 2*H[d, j, :, i]
        idx[2] += [(k, c, l, b, u, a) for u in range(n)]
        hs[2][i, :, k, b, l, a] += -2*H[c, j, d, :]
        idx[2] += [(i, u, k, b, l, a) for u in range(n)]
        hs[2][i, :, k, b, l, a] += 2*H[d, j, c, :]
        idx[2] += [(i, u, k, b, l, a) for u in range(n)]
        hs[2][i, :, k, b, l, a] += 2*H[c, :, d, j]
        idx[2] += [(i, u, k, b, l, a) for u in range(n)]
        hs[2][i, :, k, b, l, a] += -2*H[d, :, c, j]
        idx[2] += [(i, u, k, b, l, a) for u in range(n)]
        hs[2][j, :, k, b, l, a] += 2*H[c, i, d, :]
        idx[2] += [(j, u, k, b, l, a) for u in range(n)]
        hs[2][j, :, k, b, l, a] += -2*H[d, i, c, :]
        idx[2] += [(j, u, k, b, l, a) for u in range(n)]
        hs[2][j, :, k, b, l, a] += -2*H[c, :, d, i]
        idx[2] += [(j, u, k, b, l, a) for u in range(n)]
        hs[2][j, :, k, b, l, a] += 2*H[d, :, c, i]
        idx[2] += [(j, u, k, b, l, a) for u in range(n)]
        hs[1][i, d, j, c] += 2*H[a, l, b, k]
        idx[1] += [(i, d, j, c)]
        hs[1][i, d, j, c] += -2*H[a, k, b, l]
        idx[1] += [(i, d, j, c)]
        hs[2][i, :, j, d, k, c] += 2*H[a, :, b, l]
        idx[2] += [(i, u, j, d, k, c) for u in range(n)]
        hs[2][i, :, j, d, k, c] += -2*H[a, l, b, :]
        idx[2] += [(i, u, j, d, k, c) for u in range(n)]
        hs[2][i, :, j, d, l, c] += -2*H[a, :, b, k]
        idx[2] += [(i, u, j, d, l, c) for u in range(n)]
        hs[2][i, :, j, d, l, c] += 2*H[a, k, b, :]
        idx[2] += [(i, u, j, d, l, c) for u in range(n)]
        hs[1][i, d, j, c] += -2*H[b, l, a, k]
        idx[1] += [(i, d, j, c)]
        hs[1][i, d, j, c] += 2*H[b, k, a, l]
        idx[1] += [(i, d, j, c)]
        hs[2][i, :, j, d, k, c] += -2*H[b, :, a, l]
        idx[2] += [(i, u, j, d, k, c) for u in range(n)]
        hs[2][i, :, j, d, k, c] += 2*H[b, l, a, :]
        idx[2] += [(i, u, j, d, k, c) for u in range(n)]
        hs[2][i, :, j, d, l, c] += 2*H[b, :, a, k]
        idx[2] += [(i, u, j, d, l, c) for u in range(n)]
        hs[2][i, :, j, d, l, c] += -2*H[b, k, a, :]
        idx[2] += [(i, u, j, d, l, c) for u in range(n)]
        hs[2][i, d, j, c, :, b] += 2*H[:, l, a, k]
        idx[2] += [(i, d, j, c, u, b) for u in range(n)]
        hs[2][i, d, j, c, :, b] += -2*H[:, k, a, l]
        idx[2] += [(i, d, j, c, u, b) for u in range(n)]
        hs[2][i, d, j, c, :, a] += -2*H[:, l, b, k]
        idx[2] += [(i, d, j, c, u, a) for u in range(n)]
        hs[2][i, d, j, c, :, a] += 2*H[:, k, b, l]
        idx[2] += [(i, d, j, c, u, a) for u in range(n)]
        hs[2][i, d, j, c, :, b] += -2*H[a, l, :, k]
        idx[2] += [(i, d, j, c, u, b) for u in range(n)]
        hs[2][i, d, j, c, :, b] += 2*H[a, k, :, l]
        idx[2] += [(i, d, j, c, u, b) for u in range(n)]
        hs[2][i, d, j, c, :, a] += 2*H[b, l, :, k]
        idx[2] += [(i, d, j, c, u, a) for u in range(n)]
        hs[2][i, d, j, c, :, a] += -2*H[b, k, :, l]
        idx[2] += [(i, d, j, c, u, a) for u in range(n)]
        hs[3][i, :, j, d, l, c, :, a] += -2*H[:, k, b, :].transpose(1, 0)
        idx[3] += [(i, u, j, d, l, c, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[3][i, :, j, d, l, c, :, a] += 2*H[:, :, b, k].transpose(1, 0)
        idx[3] += [(i, u, j, d, l, c, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[3][i, :, j, d, k, c, :, a] += 2*H[:, l, b, :].transpose(1, 0)
        idx[3] += [(i, u, j, d, k, c, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[3][i, :, j, d, k, c, :, a] += -2*H[:, :, b, l].transpose(1, 0)
        idx[3] += [(i, u, j, d, k, c, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[3][j, d, l, c, :, b, :, a] += -2*H[:, k, :, i]
        idx[3] += [(j, d, l, c, u, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[3][j, d, k, c, :, b, :, a] += 2*H[:, l, :, i]
        idx[3] += [(j, d, k, c, u, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[3][j, :, k, c, l, b, :, a] += -2*H[d, :, :, i]
        idx[3] += [(j, u, k, c, l, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[3][j, :, k, d, l, b, :, a] += 2*H[c, :, :, i]
        idx[3] += [(j, u, k, d, l, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[3][j, :, k, c, l, b, :, a] += 2*H[:, :, d, i].transpose(1, 0)
        idx[3] += [(j, u, k, c, l, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[3][j, :, k, d, l, b, :, a] += -2*H[:, :, c, i].transpose(1, 0)
        idx[3] += [(j, u, k, d, l, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[3][i, :, j, d, l, c, :, a] += 2*H[b, k, :, :].transpose(1, 0)
        idx[3] += [(i, u, j, d, l, c, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[3][i, :, j, d, l, c, :, a] += -2*H[b, :, :, k]
        idx[3] += [(i, u, j, d, l, c, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[3][j, d, l, c, :, b, :, a] += 2*H[:, i, :, k]
        idx[3] += [(j, d, l, c, u, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[3][j, d, k, c, :, b, :, a] += -2*H[:, i, :, l]
        idx[3] += [(j, d, k, c, u, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[3][j, :, k, c, l, b, :, a] += 2*H[d, i, :, :].transpose(1, 0)
        idx[3] += [(j, u, k, c, l, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[3][j, :, k, d, l, b, :, a] += -2*H[c, i, :, :].transpose(1, 0)
        idx[3] += [(j, u, k, d, l, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[3][j, :, k, c, l, b, :, a] += -2*H[:, i, d, :].transpose(1, 0)
        idx[3] += [(j, u, k, c, l, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[3][j, :, k, d, l, b, :, a] += 2*H[:, i, c, :].transpose(1, 0)
        idx[3] += [(j, u, k, d, l, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[3][i, :, j, :, k, c, l, a] += -2*H[d, :, b, :].transpose(1, 0)
        idx[3] += [(i, u, j, v, k, c, l, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[3][i, :, j, :, k, d, l, a] += 2*H[c, :, b, :].transpose(1, 0)
        idx[3] += [(i, u, j, v, k, d, l, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[3][i, d, l, c, :, b, :, a] += 2*H[:, k, :, j]
        idx[3] += [(i, d, l, c, u, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[3][i, d, k, c, :, b, :, a] += -2*H[:, l, :, j]
        idx[3] += [(i, d, k, c, u, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[3][i, :, k, c, l, b, :, a] += 2*H[d, :, :, j]
        idx[3] += [(i, u, k, c, l, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[3][i, :, k, d, l, b, :, a] += -2*H[c, :, :, j]
        idx[3] += [(i, u, k, d, l, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[3][i, :, k, c, l, b, :, a] += -2*H[:, :, d, j].transpose(1, 0)
        idx[3] += [(i, u, k, c, l, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[3][i, :, k, d, l, b, :, a] += 2*H[:, :, c, j].transpose(1, 0)
        idx[3] += [(i, u, k, d, l, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[3][i, :, j, d, l, c, :, b] += 2*H[:, k, a, :].transpose(1, 0)
        idx[3] += [(i, u, j, d, l, c, v, b) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[3][i, :, j, d, l, c, :, b] += -2*H[:, :, a, k].transpose(1, 0)
        idx[3] += [(i, u, j, d, l, c, v, b) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[3][i, d, l, c, :, b, :, a] += -2*H[:, j, :, k]
        idx[3] += [(i, d, l, c, u, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[3][i, d, k, c, :, b, :, a] += 2*H[:, j, :, l]
        idx[3] += [(i, d, k, c, u, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[3][i, :, k, c, l, b, :, a] += -2*H[d, j, :, :].transpose(1, 0)
        idx[3] += [(i, u, k, c, l, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[3][i, :, k, d, l, b, :, a] += 2*H[c, j, :, :].transpose(1, 0)
        idx[3] += [(i, u, k, d, l, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[3][i, :, k, c, l, b, :, a] += 2*H[:, j, d, :].transpose(1, 0)
        idx[3] += [(i, u, k, c, l, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[3][i, :, k, d, l, b, :, a] += -2*H[:, j, c, :].transpose(1, 0)
        idx[3] += [(i, u, k, d, l, b, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[3][i, :, j, d, k, c, :, b] += -2*H[:, l, a, :].transpose(1, 0)
        idx[3] += [(i, u, j, d, k, c, v, b) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[3][i, :, j, d, k, c, :, b] += 2*H[:, :, a, l].transpose(1, 0)
        idx[3] += [(i, u, j, d, k, c, v, b) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[3][i, :, j, :, k, c, l, a] += 2*H[b, :, d, :].transpose(1, 0)
        idx[3] += [(i, u, j, v, k, c, l, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[3][i, :, j, :, k, d, l, a] += -2*H[b, :, c, :].transpose(1, 0)
        idx[3] += [(i, u, j, v, k, d, l, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[3][i, :, j, :, k, c, l, b] += 2*H[d, :, a, :].transpose(1, 0)
        idx[3] += [(i, u, j, v, k, c, l, b) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[3][i, :, j, :, k, d, l, b] += -2*H[c, :, a, :].transpose(1, 0)
        idx[3] += [(i, u, j, v, k, d, l, b) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[3][i, :, j, d, l, c, :, b] += -2*H[a, k, :, :].transpose(1, 0)
        idx[3] += [(i, u, j, d, l, c, v, b) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[3][i, :, j, d, l, c, :, b] += 2*H[a, :, :, k]
        idx[3] += [(i, u, j, d, l, c, v, b) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[3][i, :, j, d, k, c, :, b] += 2*H[a, l, :, :].transpose(1, 0)
        idx[3] += [(i, u, j, d, k, c, v, b) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[3][i, :, j, d, k, c, :, b] += -2*H[a, :, :, l]
        idx[3] += [(i, u, j, d, k, c, v, b) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[3][i, :, j, d, k, c, :, a] += -2*H[b, l, :, :].transpose(1, 0)
        idx[3] += [(i, u, j, d, k, c, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[3][i, :, j, d, k, c, :, a] += 2*H[b, :, :, l]
        idx[3] += [(i, u, j, d, k, c, v, a) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[3][i, :, j, :, k, c, l, b] += -2*H[a, :, d, :].transpose(1, 0)
        idx[3] += [(i, u, j, v, k, c, l, b) for (u, v) in itertools.product(range(n), repeat=2)]
        hs[3][i, :, j, :, k, d, l, b] += 2*H[a, :, c, :].transpose(1, 0)
        idx[3] += [(i, u, j, v, k, d, l, b) for (u, v) in itertools.product(range(n), repeat=2)]

    for k in range(4):
        if(idx[k] is not None):
            idx[k] = list(set(idx[k]))

    return FermionicOperatorNBody(hs), idx
