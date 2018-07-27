#!/usr/bin/env python3

import numpy as np
import scipy

from qiskit_aqua.utils import tensorproduct


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
    Generate a random unitary matrix with NxN matrix.

    Args:
        N: the dimension of unitary matrixs
    Returns:
        np.ndarray: a 2-D matrix with np.complex data type.
    """
    X = (np.random.randint(N, size=(N, N)) + 1j*np.random.randint(N, size=(N, N))) / np.sqrt(2)
    Q, R = np.linalg.qr(X)
    R = np.diag(np.multiply(np.diag(R), abs(np.diag(R))))
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
        H2BodyS[Htemp[i, 0] + N//2, Htemp[i, 1] + N//2, Htemp[i, 2] + N//2, Htemp[i, 3] + N//2] = val[i]
        H2BodyS[Htemp[i, 0] + N//2, Htemp[i, 1] + N//2, Htemp[i, 2],
                Htemp[i, 3]] = val[i]  # shift i and j to their spin symmetrized
        H2BodyS[Htemp[i, 0], Htemp[i, 1], Htemp[i, 2] + N//2, Htemp[i, 3] +
                N//2] = val[i]  # shift l and m to their spin symmetrized
    return H2BodyS
