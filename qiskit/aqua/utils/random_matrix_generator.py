#!/usr/bin/env python3

import numpy as np
import scipy

from qiskit.aqua.utils import tensorproduct


def random_h1_body(N):
    """
    Generate a random one body integrals.

    Args:
        N (int): the number of spin orbitals.

    Returns:
        np.ndarray: a 2-D matrix with np.complex data type.
    """
    Pup = np.asarray([[1, 0], [0, 0]])
    Pdown = np.asarray([[0, 0], [0, 1]])

    if N % 2 != 0:
        raise ValueError('The number of spin-orbitals must be even but {}'.format(N))
    h1 = np.ones((N // 2, N // 2)) - 2 * np.random.random((N // 2, N // 2))
    h1 = np.triu(tensorproduct(Pup, h1) + tensorproduct(Pdown, h1))
    h1 = (h1 + h1.T) / 2.0
    return h1


def random_unitary(N):
    """
    Generate a random unitary matrix with size NxN.

    Args:
        N: the dimension of unitary matrix
    Returns:
        np.ndarray: a 2-D matrix with np.complex data type.
    """
    X = (np.random.random(size=(N, N))*N + 1j*np.random.random(size=(N, N))*N) / np.sqrt(2)
    Q, R = np.linalg.qr(X)
    R = np.diag(np.divide(np.diag(R), abs(np.diag(R))))
    unitary_matrix = np.dot(Q, R)
    return unitary_matrix


def random_h2_body(N, M):
    """
    Generate a random two body integrals.
    Args:
        N (int) : number of spin-orbitals (dimentsion of h2)
        M (int) : number of non-zero entries
    Returns:
        np.ndarray: a numpy 4-D tensor with np.complex data type.
    """
    if N % 2 == 1:
        assert 0, "The number of spin orbitals must be even."

    h2 = np.zeros((N//2, N//2, N//2, N//2))
    max_nonzero_elements = 0

    if N / 2 != 1:
        if N / 2 >= 2:
            max_nonzero_elements += 4 * 4 * scipy.special.comb(N//2, 2)
        if N / 2 >= 3:
            max_nonzero_elements += 4 * 3 * 8 * scipy.special.comb(N//2, 3)
        if N / 2 >= 4:
            max_nonzero_elements += 4 * scipy.special.factorial(N//2) / scipy.special.factorial(N//2-4)
        #print('Max number of non-zero elements for {} spin-orbitals is: {}'.format(N, max_nonzero_elements))

    if M > max_nonzero_elements:
        assert 0, 'Too many non-zero elements required, given the molecular symmetries. \n\
                    The maximal number of non-zero elements for {} spin-orbitals is {}'.format(N, max_nonzero_elements)

    element_count = 0
    while element_count < M:
        r_i = np.random.randint(N // 2, size=(4))
        i, j, l, m = r_i[0], r_i[1], r_i[2], r_i[3]
        if i != l and j != m and h2[i, j, l, m] == 0:
            h2[i, j, l, m] = 1 - 2 * np.random.random(1)
            element_count += 4
            # In the chemists notation H2BodyS(i,j,l,m) refers to
            # a^dag_i a^dag_l a_m a_j
            if h2[l, m, i, j] == 0:
                h2[l, m, i, j] = h2[i, j, l, m]
                element_count += 4
            if h2[j, i, m, l] == 0:
                h2[j, i, m, l] = h2[i, j, l, m]
                element_count += 4
            if h2[m, l, j, i] == 0:
                h2[m, l, j, i] = h2[i, j, l, m]
                element_count += 4
            if j != l and i != m:
                # if these conditions are not satisfied the symmetries
                # will produce (negligible) terms that annihilate any state
                if h2[j, i, l, m] == 0:
                    h2[j, i, l, m] = h2[i, j, l, m]
                    element_count += 4
                if h2[m, l, i, j] == 0:
                    h2[m, l, i, j] = h2[i, j, l, m]
                    element_count += 4
                if h2[i, j, m, l] == 0:
                    h2[i, j, m, l] = h2[i, j, l, m]
                    element_count += 4
                if h2[l, m, j, i] == 0:
                    h2[l, m, j, i] = h2[i, j, l, m]
                    element_count += 4

    # Impose spin degeneracy
    idx_non_zeros = np.where(h2 != 0)
    a, b, c, d = idx_non_zeros[0], idx_non_zeros[1], idx_non_zeros[2], idx_non_zeros[3]
    val = h2[idx_non_zeros]
    Htemp = np.column_stack((a, b, c, d)).astype(int)

    dim = Htemp.shape
    H2BodyS = np.zeros((N, N, N, N))
    H2BodyS[0:N//2, 0:N//2, 0:N//2, 0:N//2] = h2
    for i in range(dim[0]):
        # recall that in the chemists notation H2BodyS(i,j,l,m) refers to
        # a^dag_i a^dag_l a_m a_j
        H2BodyS[Htemp[i, 0] + N//2, Htemp[i, 1] + N//2, Htemp[i, 2] + N//2,
                Htemp[i, 3] + N//2] = val[i]
        H2BodyS[Htemp[i, 0] + N//2, Htemp[i, 1] + N//2, Htemp[i, 2],
                Htemp[i, 3]] = val[i]  # shift i and j to their spin symmetrized
        H2BodyS[Htemp[i, 0], Htemp[i, 1], Htemp[i, 2] + N//2, Htemp[i, 3] +
                N//2] = val[i]  # shift l and m to their spin symmetrized
    return H2BodyS


def random_diag(N, eigs=None, K=None, eigrange=[0, 1]):
    """
    Generate random diagonal matix with given properties
    Args:
        N (int): size of matrix
        eigs (list, tuple, np.ndarray): list of N eigenvalues. Overrides K,
                                        eigrange.
        K (float, list, tuple): condition number. Either use only condition
                                number K or list/tuple of (K, lmin) or (K,
                                lmin, sgn). Where lmin is the smallest
                                eigenvalue and sign +/- 1 specifies if
                                eigenvalues can be negative.
        eigrange (list, tuple, nd.ndarray): [min, max] list for eigenvalue
                                            range. (default=[0, 1])
    Returns:
        np.ndarray: diagonal matrix
    """
    if not isinstance(eigs, np.ndarray):
        if eigs is None:
            if not isinstance(K, np.ndarray) and K is not None:
                if isinstance(K, (int, float)):
                    k, lmin, sgn = K, 1, 1
                elif len(K) == 2:
                    k, lmin = K
                    sgn = 1
                elif len(K) == 3:
                    k, lmin, sgn = K
                eigs = np.random.random(N)
                a = (k-1)*lmin/(max(eigs)-min(eigs))
                b = lmin*(max(eigs)-k*min(eigs))/(max(eigs)-min(eigs))
                eigs = a*eigs+b
                if sgn == -1:
                    sgs = np.random.random(N)-0.5
                    while min(sgs) > 0 or max(sgs) < 0:
                        sgs = np.random.random(N)-0.5
                    eigs = eigs*(sgs/abs(sgs))
            elif isinstance(eigrange, (tuple, list, np.ndarray)) \
                    and len(eigrange) == 2:
                eigs = np.random.random(N) * \
                       (eigrange[1]-eigrange[0])+eigrange[0]
            else:
                raise ValueError("Wrong input data: either 'eigs', 'K' or"
                                 "'eigrange' needed to be set correctly.")
        else:
            assert len(eigs) == N, "NxN matrix needs N eigenvalues."
            eigs = np.array(list(eigs))
    else:
        assert len(eigs) == N, "NxN matrix needs N eigenvalues."
    return np.diag(eigs)


def limit_paulis(mat, n=5, sparsity=None):
    """
    Limits the number of Pauli basis matrices of a hermitian matrix to the n
    highest magnitude ones.
    Args:
        mat (np.ndarray): Input matrix
        n (int): number of surviving Pauli matrices (default=5)
        sparsity (float < 1): sparsity of matrix

    Returns:
        scipy.sparse.csr_matrix
    """
    from qiskit.aqua import Operator
    # Bringing matrix into form 2**Nx2**N
    l = mat.shape[0]
    if np.log2(l) % 1 != 0:
        k = int(2**np.ceil(np.log2(l)))
        m = np.zeros([k, k], dtype=np.complex128)
        m[:l, :l] = mat
        m[l:, l:] = np.identity(k-l)
        mat = m

    # Getting Pauli matrices
    op = Operator(matrix=mat)
    op._check_representation("paulis")
    op._simplify_paulis()
    paulis = sorted(op.paulis, key=lambda x: abs(x[0]), reverse=True)
    g = 2**op.num_qubits
    mat = scipy.sparse.csr_matrix(([], ([], [])), shape=(g, g),
                                  dtype=np.complex128)

    # Truncation
    if sparsity is None:
        for pa in paulis[:n]:
            mat += pa[0]*pa[1].to_spmatrix()
    else:
        idx = 0
        while mat[:l, :l].nnz/l**2 < sparsity:
            mat += paulis[idx][0]*paulis[idx][1].to_spmatrix()
            idx += 1
        n = idx
    mat = mat.toarray()
    return mat[:l, :l]


def random_hermitian(N, eigs=None, K=None, eigrange=[0, 1], sparsity=None,
                     trunc=None):
    """
    Generate random hermitian (sparse) matix with given properties. Sparsity is
    achieved by truncating Pauli matrices. Sparsity settings alternate the
    eigenvalues due to truncation.
    Args:
        N (int): size of matrix
        eigs (list, tuple, np.ndarray): list of N eigenvalues. Overrides K,
                                        eigrange
        K (float, list, tuple): condition number. Either use only condition
                                number K or list/tuple of (K, lmin) or (K,
                                lmin, sgn). Where lmin is the smallest
                                eigenvalue and sign +/- 1 specifies if
                                eigenvalues can be negative.
        eigrange (list, tuple, nd.ndarray): [min, max] list for eigenvalue
                                            range. (default=[0, 1])
        trunc (int): limit for number of Pauli matrices.
        sparsity (float): sparsity of matrix. Overrides trunc.
    Returns:
        np.ndarray: hermitian matrix
    """
    if N == 1:
        raise ValueError('The matrix dimension must be larger than 1')
    u = scipy.stats.unitary_group.rvs(N)
    d = random_diag(N, eigs, K, eigrange)
    ret = u.conj().T.dot(d).dot(u)
    if sparsity or trunc:
        ret = limit_paulis(ret, trunc, sparsity)
    return ret


def limit_entries(mat, n=5, sparsity=None):
    """
    Limits the number of entries of a matrix to the n highest magnitude ones.
    Args:
        mat (np.array): Input Matrix
        n (int): number of surviving entries (default=5)
        sparsity (float < 1): sparsity of matrix

    Returns:
        scipy.sparse.csr_matrix
    """
    ret = scipy.sparse.coo_matrix(mat)
    entries = list(sorted(zip(ret.row, ret.col, ret.data),
                          key=lambda x: abs(x[2]), reverse=True))
    if sparsity is not None:
        n = int(sparsity*mat.shape[0]*mat.shape[1])
    entries = entries[:n]
    row, col, data = np.array(entries).T
    return scipy.sparse.csr_matrix(
        (data, (row.real.astype(int), col.real.astype(int))))


def random_non_hermitian(N, M=None, sings=None, K=None, srange=[0, 1],
                         sparsity=None, trunc=None):
    """
    Generate random (sparse) matrix with given properties (singular values).
    Sparsity is achieved by truncating Pauli matrices. Sparsity settings
    alternate the singular values due to truncation.
    Args:
        N (int): size of matrix
        sings (list, tuple, np.ndarray): list of N singular values.
                                         Overrides K, srange.
        K (float, list, tuple): condition number. Either use only condition
                                number K or list/tuple of (K, lmin). Where lmin
                                specifies the smallest singular value.
        srange (list, tuple, nd.ndarray): [min, max] list for singular value
                                          range, min >= 0. (default=[0, 1]).
        trunc (int): limit of Pauli matrices.
        sparsity (float): sparsity of matrix. Overrides trunc.
    Returns:
        np.ndarray: random matrix
    """
    if N == 1:
        raise ValueError('The matrix dimension must be larger than 1')
    if M is None:
        M = N
    d = random_diag(min(N, M), sings, K, srange)
    u = scipy.stats.unitary_group.rvs(M)
    v = scipy.stats.unitary_group.rvs(N)
    if M > N:
        s = np.zeros([M, N])
        s[:N, :] = d
    elif N > M:
        s = np.zeros([M, N])
        s[:, :M] = d
    else:
        s = d
    ret = u.dot(s).dot(v.T.conj())
    if sparsity or trunc:
        ret = limit_entries(ret, trunc, sparsity).toarray()
    return ret
