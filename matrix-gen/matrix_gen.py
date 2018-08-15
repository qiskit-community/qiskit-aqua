import numpy as np
import random
import scipy.sparse as sparse
from scipy.sparse.linalg import inv, eigs, expm, eigsh
from numpy.linalg import eig, eigh
from scipy.stats import unitary_group
from qiskit_aqua import Operator

pauli = [np.array([[1, 0], [0, 1]]), np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]])]


def _gen_pauli(n):
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
    v, u = eigh(_gen_sparse_hermitian(n, sparsity=sparsity).toarray())
    while not check_unitary(u):
        #print(v, u)
        v, u = eigh(_gen_sparse_hermitian(n, sparsity=sparsity).toarray())
    #u = (abs(u) > 10**-14)*u
    print(check_unitary(u))
    return sparse.csr_matrix(u)


def _gen_unitary(n, d):
    ps = np.array([_gen_pauli(n) for _ in range(d)])
    fs = np.random.random(d)
    h = sum([f*p for f, p in zip(fs, ps)])

def _gen_random_unitary(n):
    return unitary_group.rvs(n)

def _gen_regular_sparse(n, sparsity=0.01):
    invertible = False
    while not invertible:
        s = sparse.random(n, n, density=sparsity)
        sdag = inv(s)
        invertible = True
    return s, sdag

def _gen_random_diag(n, lminmax=None, K=None, lmin=1, eigs=None):
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
    u = _gen_random_unitary(n)
    d = _gen_random_diag(n, K=K, lminmax=lminmax, lmin=lmin, eigs=eigs)
    return u.dot(d).dot(u.T.conj())

def _gen_random_hermitian_sparse_evals(n, sparsity=0.1, lminmax=None, K=None, lmin=1, eigs=None):
    u = _gen_sparse_unitary(n).toarray()
    d = _gen_random_diag(n, K=K, lminmax=lminmax, lmin=lmin, eigs=eigs)
    return u.dot(d).dot(u.T.conj())

def check_hermitian(mat):
    return sum(sum(abs(mat-mat.T.conj()))) < 10**-13*mat.shape[0]*mat.shape[1]

def check_unitary(mat):
    x = sum(sum(abs(mat.dot(mat.T.conj())-np.identity(mat.shape[0]))))
    print(x)
    return x < 10**-12*mat.shape[0]*mat.shape[1]

def limit_paulis(mat, n):
    if np.log2(mat.shape[0]) % 1 != 0:
        print("fail")
        return
    op = Operator(matrix=mat)
    op._check_representation("paulis")
    op._simplify_paulis()
    paulis = op.paulis
    p = len(paulis)
    paulis = sorted(op.paulis, key=lambda x: abs(x[0]), reverse=True)[:n]
    #paulis = list(map(lambda c: (abs(c[0]), c[1]), paulis))
    op = Operator(paulis=paulis)
    op._simplify_paulis()
    op._paulis_to_matrix()
    print("reduction factor:", n/p)
    return op._matrix


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    #print(_gen_random_hermitian_evals(4, eigs=np.linspace(1, 10, 4)))
    evals = []
    N = 5
    k = 3
    q = 0
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
#h = _gen_pauli_hermitian(2, 4, scale=10, negative=True)

#print(check_hermitian(h))
#print(eig(h)[0].real)
