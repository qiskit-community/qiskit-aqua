import numpy as np
import random
import scipy.sparse as sparse
from scipy.sparse.linalg import inv, eigs, expm, eigsh
from numpy.linalg import eig, eigh, svd
from scipy.stats import unitary_group
from qiskit_aqua import Operator

"""
Module for generating random testing matrices.
Use only gen_matrix.
"""

def _gen_pauli(n):
    pauli = [np.array([[1, 0], [0, 1]]), np.array([[0, 1], [1, 0]]),
                np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]])]
    ps = [random.choice(list(enumerate(pauli))) for _ in range(n)]
    x = np.array([[1]])
    s = ""
    for i, p in ps:
        x = np.kron(x, p)
        s += str(i)
    return x, s

def _gen_pauli_hermitian(n, d, scale=1, negative=False):
    keys = []
    ret = np.zeros((2**n, 2**n))
    if negative:
        offset = -scale
        scale *= 2
    else:
        offset = 0
    while len(keys) < d:
        p, k = _gen_pauli(n)
        if k not in keys:
            keys.append(k)
            f = random.random()*scale+offset
            print(f)
            ret = ret + f*p
    return ret

def _gen_sparse_hermitian(n, sparsity=0.01):
    h = sparse.triu(sparse.random(n, n, density=sparsity))
    cl = 1j*np.random.random(h.nnz)
    h += sparse.csc_matrix((cl, (h.row, h.col)), shape=(n, n))
    h = h+h.conj().T
    return h.tocsc()

def _gen_sparse_unitary(n, sparsity=0.01):
    #print(check_hermitian(_gen_sparse_hermitian(n, sparsity=sparsity).toarray()))
    sh = _gen_sparse_hermitian(n, sparsity=sparsity).toarray()
    v, u = eigh(sh)
    while not check_unitary(u):
        #print(v, u)
        sh = _gen_sparse_hermitian(n, sparsity=sparsity).toarray()
        v, u = eigh(sh)
    #u = (abs(u) > 10**-14)*u
    print(sh)
    return sparse.csr_matrix(u)

def _gen_random_unitary(n):
    """
    generates random unitary matrix.
    """
    return unitary_group.rvs(n)

def _gen_regular_sparse(n, sparsity=0.01):
    invertible = False
    while not invertible:
        s = sparse.random(n, n, density=sparsity)
        sdag = inv(s)
        invertible = True
    return s, sdag

def _gen_random_diag(n, lminmax=None, K=None, lmin=1, eigs=None):
    """
    generates random diagonal matrix with specified eigenvalue conditions.
    """
    if K:
        lmax = K*lmin
    elif lminmax:
        lmin, lmax = lminmax
    else:
        lmax=1
        lmin=0
    vec = lmin+np.random.random(n)*(lmax-lmin) if np.any(eigs) == None else eigs
    print(vec)
    return np.diag(vec)

def _gen_random_hermitian_evals(n, lminmax=None, K=None, lmin=1, eigs=None):
    """
    generates random hermitian matrix with specified eigenvalues
    """
    u = _gen_random_unitary(n)
    d = _gen_random_diag(n, K=K, lminmax=lminmax, lmin=lmin, eigs=eigs)
    return u.dot(d).dot(u.T.conj())

def _gen_random_hermitian_sparse_evals(n, sparsity=0.1, lminmax=None, K=None, lmin=1, eigs=None):
    u = _gen_sparse_unitary(n).toarray()
    print(u)
    d = _gen_random_diag(n, K=K, lminmax=lminmax, lmin=lmin, eigs=eigs)
    return u.dot(d).dot(u.T.conj())

def _plot_pauli_reduction(s, k, N=5):
    import matplotlib.pyplot as plt
    q = 0
    evals = []
    while q < N:
        input = _gen_random_hermitian_evals(2**k, eigs=np.arange(1, 2**k+1))
        ref = eigh(input)[0]
        x = range(2, 4**k, 4)
        eval = [[] for _ in ref]
        for i in x:
            mat = limit_paulis(input, i)
            if mat == None:
                q -= 1
                break
            for i, e in enumerate(eigh(mat.toarray())[0]):
                eval[i].append(e)
        evals.append(eval)
        q += 1

    feval = []
    for i in range(len(ref)):
        x = np.zeros(len(evals[0][i]))
        for eval in evals:
            eval[i] = np.array(eval[i])
            x = x+eval[i]
        feval.append(x/N)

    for e in feval:
        plt.plot(x, e)
    for r in ref:
        plt.axhline(y=r, color="black")
    plt.show()

def _gen_random_triu_sparse(n, sparsity=0.1, diag_ones=True):
    N = int(sparsity*n**2)
    rows = []
    cols = []
    data = []
    if diag_ones:
        N = int(max(N-n, 0))
        for i in range(n):
            rows.append(i)
            cols.append(i)
            data.append(1)
    for _ in range(N):
        j = np.random.randint(1, n)
        cols.append(j)
        rows.append(np.random.randint(0, j))
        data.append(np.random.random(1)+1j*np.random.random(1))
    return sparse.csr_matrix((data, (rows, cols)), shape=(n, n))

def _gen_random_evals(n, m=None, sminmax=None, K=None, sings=None, smin=1):
    """
    generates random non-hermitian matrix with specified singular values
    """
    if m == None:
        m = n
    d = _gen_random_diag(min(m, n), K=K, lminmax=sminmax, lmin=smin, eigs=sings)
    u = _gen_random_unitary(m)
    v = _gen_random_unitary(n)
    if m > n:
        s = np.zeros([m, n])
        s[:n,:] = d
    elif n > m:
        s = np.zeros([m, n])
        s[:,:m] = d
    else:
        s = d
    return u.dot(s).dot(v.T.conj())

def check_hermitian(mat):
    """
    checks if matrix is hermitian (relative tolerance)

    Args:
        mat (np.array): Input matrix

    Output:
        boolean
    """
    return sum(sum(abs(mat-mat.T.conj()))) < 10**-13*mat.shape[0]*mat.shape[1]

def check_unitary(mat):
    """
    checks unitarity of matrix (relative tolerance)

    Args:
        mat (np.array): Input matrix

    Output:
        boolean
    """
    x = sum(sum(abs(mat.dot(mat.T.conj())-np.identity(mat.shape[0]))))
    return x < 10**-12*mat.shape[0]*mat.shape[1]

def limit_paulis(mat, n=5, sparsity=None):
    """
    Limits the number of pauli basis matrices of a hermitian matrix to the n
    highest magnitude ones.

    Args:
        mat (np.array): Input Matrix
        n (int): number of surviving paulis (default=5)
        sparsity (float < 1): sparsity of matrix

    Outputs:
        np.sparse.csr_matrix
    """
    l = mat.shape[0]
    if np.log2(l) % 1 != 0:
        k = int(2**np.ceil(np.log2(l)))
        m = np.zeros([k, k], dtype=np.complex128)
        m[:-(k-l),:-(k-l)] = mat
        m[(l):,(l):] = np.identity(k-l)
        mat = m
    op = Operator(matrix=mat)
    op._check_representation("paulis")
    op._simplify_paulis()
    paulis = op.paulis
    p = len(paulis)
    paulis = sorted(op.paulis, key=lambda x: abs(x[0]), reverse=True)
    #paulis = list(map(lambda c: (abs(c[0]), c[1]), paulis))
    g = 2**op.num_qubits
    mat = sparse.csr_matrix(([], ([], [])), shape=(g, g), dtype=np.complex128)
    if sparsity==None:
        for pa in paulis[:n]:
            mat += pa[0]*pa[1].to_spmatrix()
    else:
        idx = 0
        while mat.nnz/g**2 < sparsity:
            mat += paulis[idx][0]*paulis[idx][1].to_spmatrix()
            idx += 1
        n = idx
    print("reduction factor:", n/p)
    if np.log2(l) % 1 != 0:
        mat = mat.toarray()
        mat = mat[:-(k-l), :-(k-l)]
    return sparse.csr_matrix(mat)

def limit_entries(mat, n=5, sparsity=None):
    """
    Limits the number of entries of a matrix to the n highest magnitude ones.

    Args:
        mat (np.array): Input Matrix
        n (int): number of surviving entries (default=5)
        sparsity (float < 1): sparsity of matrix

    Outputs:
        np.sparse.csr_matrix
    """
    ret = sparse.coo_matrix(mat)
    entries = list(sorted(zip(ret.row, ret.col, ret.data), key=lambda x: abs(x[2]), reverse=True))
    if sparsity != None:
        n = int(sparsity*mat.shape[0]*mat.shape[1])
    entries = entries[:n]
    row, col, data = np.array(entries).T
    return sparse.csr_matrix((data, (row.real.astype(int), col.real.astype(int))))

def gen_matrix(n, m=None, hermitian=True, sparsity=None, condition=None,
                eigrange=None, eigvals=None, trunc=None):
    """
    A method for generating (sparse, hermitian) random matrices with given
    eigenvalues or condition number, resp. The method of sparsifying the matrix
    involves truncation of the pauli base matrices (to those with the largest
    prefactors) in the hermitian case and truncation of the smallest magnitude
    matrix entries in the non-hermitian case. Therefore sparsified matrices do
    not fully have the specified eignevals or condition number, but one quite
    close to them, dpending on the magnitude of truncation.

    Args:
        n (int): Size of the (square) matrix
        m (int): Additional dimenstion size for non-square m x n matrices
                 (optional)
        hermitian (bool): Whether output matrix should be hermitian or not
                          (default=True)
        sparsity (float < 1): Degree of sparsity non-zero-elements/n*m
                              (optional, no effect if hermitian is set)
        condition (float): Condition number of the matrix smalles eigen-/
                           singularvalue is set to 1 (optional)
        eigrange (list(float)): Lower and upper bound of randomly generated
                                eigen-/singularvalues (optional, no effect if
                                condition is set)
        eigvals (list(float)): All specific eigen-/singlularvalues (length:
                               min(n, m)) (optional, no effect if condition or
                               eigrange is set)
        tunc (int): number of largest value paulis/entries that are used in the
                    output matrix (optional, no effect if sparsity is set)

    Output:
        np.array
    """
    if hermitian:
        mat = _gen_random_hermitian_evals(n, lminmax=eigrange, K=condition, eigs=eigvals)
        if sparsity or trunc:
            mat = limit_paulis(mat, n=trunc, sparsity=sparsity).toarray()
    else:
        mat = _gen_random_evals(n, m, lminmax=eigrange, K=condition, eigs=eigvals)
        if sparsity or trunc:
            mat = limit_entries(mat, n=trunc, sparsity=sparsity).toarray()
    return mat
