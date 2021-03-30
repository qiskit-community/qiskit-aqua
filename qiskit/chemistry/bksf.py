# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" BKSF methods """

import copy
import itertools

import retworkx
import numpy as np
from qiskit.quantum_info import Pauli
from qiskit.aqua.operators import WeightedPauliOperator


def _one_body(edge_list, p, q, h1_pq):  # pylint: disable=invalid-name
    """
    Map the term a^\\dagger_p a_q + a^\\dagger_q a_p to qubit operator.
    Args:
        edge_list (numpy.ndarray): 2xE matrix, each indicates (from, to) pair
        p (int): index of the one body term
        q (int): index of the one body term
        h1_pq (complex): coefficient of the one body term at (p, q)

    Return:
        WeightedPauliOperator: mapped qubit operator
    """
    # Handle off-diagonal terms.
    if p != q:
        a_i, b_i = sorted([p, q])
        b_a = edge_operator_bi(edge_list, a_i)
        b_b = edge_operator_bi(edge_list, b_i)
        a_ab = edge_operator_aij(edge_list, a_i, b_i)
        qubit_op = a_ab * b_b + b_a * a_ab
        final_coeff = -1j * 0.5

    # Handle diagonal terms.
    else:
        b_p = edge_operator_bi(edge_list, p)
        v = np.zeros(edge_list.shape[1])
        w = np.zeros(edge_list.shape[1])
        id_pauli = Pauli((v, w))

        id_op = WeightedPauliOperator(paulis=[[1.0, id_pauli]])
        qubit_op = id_op - b_p
        final_coeff = 0.5

    qubit_op = (final_coeff * h1_pq) * qubit_op
    qubit_op.simplify()
    return qubit_op


def _two_body(edge_list, p, q, r, s, h2_pqrs):  # pylint: disable=invalid-name
    """
    Map the term a^\\dagger_p a^\\dagger_q a_r a_s + h.c. to qubit operator.

    Args:
        edge_list (numpy.ndarray): 2xE matrix, each indicates (from, to) pair
        p (int): index of the two body term
        q (int): index of the two body term
        r (int): index of the two body term
        s (int): index of the two body term
        h2_pqrs (complex): coefficient of the two body term at (p, q, r, s)

    Returns:
        WeightedPauliOperator: mapped qubit operator
    """
    # Handle case of four unique indices.
    v = np.zeros(edge_list.shape[1])
    id_op = WeightedPauliOperator(paulis=[[1, Pauli((v, v))]])
    final_coeff = 1.0

    if len(set([p, q, r, s])) == 4:
        b_p = edge_operator_bi(edge_list, p)
        b_q = edge_operator_bi(edge_list, q)
        b_r = edge_operator_bi(edge_list, r)
        b_s = edge_operator_bi(edge_list, s)
        a_pq = edge_operator_aij(edge_list, p, q)
        a_rs = edge_operator_aij(edge_list, r, s)
        a_pq = -a_pq if q < p else a_pq
        a_rs = -a_rs if s < r else a_rs

        qubit_op = (a_pq * a_rs) * (-id_op - b_p * b_q + b_p * b_r
                                    + b_p * b_s + b_q * b_r + b_q * b_s
                                    - b_r * b_s - b_p * b_q * b_r * b_s)
        final_coeff = 0.125

    # Handle case of three unique indices.
    elif len(set([p, q, r, s])) == 3:
        b_p = edge_operator_bi(edge_list, p)
        b_q = edge_operator_bi(edge_list, q)
        if p == r:
            b_s = edge_operator_bi(edge_list, s)
            a_qs = edge_operator_aij(edge_list, q, s)
            a_qs = -a_qs if s < q else a_qs
            qubit_op = (a_qs * b_s + b_q * a_qs) * (id_op - b_p)
            final_coeff = 1j * 0.25
        elif p == s:
            b_r = edge_operator_bi(edge_list, r)
            a_qr = edge_operator_aij(edge_list, q, r)
            a_qr = -a_qr if r < q else a_qr
            qubit_op = (a_qr * b_r + b_q * a_qr) * (id_op - b_p)
            final_coeff = 1j * -0.25
        elif q == r:
            b_s = edge_operator_bi(edge_list, s)
            a_ps = edge_operator_aij(edge_list, p, s)
            a_ps = -a_ps if s < p else a_ps
            qubit_op = (a_ps * b_s + b_p * a_ps) * (id_op - b_q)
            final_coeff = 1j * -0.25
        elif q == s:
            b_r = edge_operator_bi(edge_list, r)
            a_pr = edge_operator_aij(edge_list, p, r)
            a_pr = -a_pr if r < p else a_pr
            qubit_op = (a_pr * b_r + b_p * a_pr) * (id_op - b_q)
            final_coeff = 1j * 0.25
        else:
            pass

    # Handle case of two unique indices.
    elif len(set([p, q, r, s])) == 2:
        b_p = edge_operator_bi(edge_list, p)
        b_q = edge_operator_bi(edge_list, q)
        qubit_op = (id_op - b_p) * (id_op - b_q)
        if p == s:
            final_coeff = 0.25
        else:
            final_coeff = -0.25
    else:
        pass

    qubit_op = (final_coeff * h2_pqrs) * qubit_op
    qubit_op.simplify()
    return qubit_op


def bravyi_kitaev_fast_edge_list(fer_op):
    """
    Construct edge list required for the bksf algorithm.

    Args:
        fer_op (FeriomicOperator): the fermionic operator in the second quantized form

    Returns:
        numpy.ndarray: edge_list, a 2xE matrix, where E is total number of edge
                        and each pair denotes (from, to)
    """
    h_1 = fer_op.h1
    h_2 = fer_op.h2
    modes = fer_op.modes
    edge_matrix = np.zeros((modes, modes), dtype=bool)

    for p, q in itertools.product(range(modes), repeat=2):  # pylint: disable=invalid-name

        if h_1[p, q] != 0.0 and p >= q:
            edge_matrix[p, q] = True

        for r, s in itertools.product(range(modes), repeat=2):  # pylint: disable=invalid-name
            if h_2[p, q, r, s] == 0.0:  # skip zero terms
                continue

            # Identify and skip one of the complex conjugates.
            if [p, q, r, s] != [s, r, q, p]:
                if len(set([p, q, r, s])) == 4:
                    if min(r, s) < min(p, q):
                        continue
                elif p != r and q < p:
                    continue

            # Handle case of four unique indices.
            if len(set([p, q, r, s])) == 4:
                if p >= q:
                    edge_matrix[p, q] = True
                    a_i, b = sorted([r, s])
                    edge_matrix[b, a_i] = True

            # Handle case of three unique indices.
            elif len(set([p, q, r, s])) == 3:
                # Identify equal tensor factors.
                if p == r:
                    a_i, b = sorted([q, s])
                elif p == s:
                    a_i, b = sorted([q, r])
                elif q == r:
                    a_i, b = sorted([p, s])
                elif q == s:
                    a_i, b = sorted([p, r])
                else:
                    continue
                edge_matrix[b, a_i] = True

    edge_list = np.asarray(np.nonzero(np.triu(edge_matrix.T) ^ np.diag(np.diag(edge_matrix.T))))
    return edge_list


def edge_operator_aij(edge_list, i, j):
    """Calculate the edge operator A_ij.

    The definitions used here are consistent with arXiv:quant-ph/0003137

    Args:
        edge_list (numpy.ndarray): a 2xE matrix, where E is total number of edge
                                    and each pair denotes (from, to)
        i (int): specifying the edge operator A
        j (int): specifying the edge operator A

    Returns:
        WeightedPauliOperator: qubit operator
    """
    v = np.zeros(edge_list.shape[1])
    w = np.zeros(edge_list.shape[1])

    position_ij = -1
    qubit_position_i = np.asarray(np.where(edge_list == i))

    for edge_index in range(edge_list.shape[1]):
        if set((i, j)) == set(edge_list[:, edge_index]):
            position_ij = edge_index
            break

    w[position_ij] = 1

    for edge_index in range(qubit_position_i.shape[1]):
        i_i, j_j = qubit_position_i[:, edge_index]
        i_i = 1 if i_i == 0 else 0  # int(not(i_i))
        if edge_list[i_i][j_j] < j:
            v[j_j] = 1

    qubit_position_j = np.asarray(np.where(edge_list == j))
    for edge_index in range(qubit_position_j.shape[1]):
        i_i, j_j = qubit_position_j[:, edge_index]
        i_i = 1 if i_i == 0 else 0  # int(not(i_i))
        if edge_list[i_i][j_j] < i:
            v[j_j] = 1

    qubit_op = WeightedPauliOperator(paulis=[[1.0, Pauli((v, w))]])
    return qubit_op


def stabilizers(fer_op):
    """ stabilizers """
    edge_list = bravyi_kitaev_fast_edge_list(fer_op)
    num_qubits = edge_list.shape[1]

    graph = retworkx.PyGraph()
    graph.extend_from_edge_list(list(map(tuple, edge_list.transpose())))
    stabs = np.asarray(retworkx.cycle_basis(graph))
    stabilizer_ops = []
    for stab in stabs:
        a_op = WeightedPauliOperator(paulis=[[1.0, Pauli.from_label('I' * num_qubits)]])
        stab = np.asarray(stab)
        for i in range(np.size(stab)):
            a_op = a_op * edge_operator_aij(edge_list, stab[i], stab[(i + 1) % np.size(stab)]) * 1j
        stabilizer_ops.append(a_op)

    return stabilizer_ops


def edge_operator_bi(edge_list, i):
    """Calculate the edge operator B_i.

    The definitions used here are consistent with arXiv:quant-ph/0003137

    Args:
        edge_list (numpy.ndarray): a 2xE matrix, where E is total number of edge
                                    and each pair denotes (from, to)
        i (int): index for specifying the edge operator B.

    Returns:
        WeightedPauliOperator: qubit operator
    """
    qubit_position_matrix = np.asarray(np.where(edge_list == i))
    qubit_position = qubit_position_matrix[1]
    v = np.zeros(edge_list.shape[1])
    w = np.zeros(edge_list.shape[1])
    v[qubit_position] = 1
    qubit_op = WeightedPauliOperator(paulis=[[1.0, Pauli((v, w))]])
    return qubit_op


def bksf_mapping(fer_op):
    """
    Transform from InteractionOpeator to QubitOperator for Bravyi-Kitaev fast algorithm.

    The electronic Hamiltonian is represented in terms of creation and
    annihilation operators. These creation and annihilation operators could be
    used to define Majorana modes as follows:
        c_{2i} = a_i + a^{\\dagger}_i,
        c_{2i+1} = (a_i - a^{\\dagger}_{i})/(1j)
    These Majorana modes can be used to define edge operators B_i and A_{ij}:
        B_i=c_{2i}c_{2i+1},
        A_{ij}=c_{2i}c_{2j}
    using these edge operators the fermionic algebra can be generated and
    hence all the terms in the electronic Hamiltonian can be expressed in
    terms of edge operators. The terms in electronic Hamiltonian can be
    divided into five types (arXiv 1208.5986). We can find the edge operator
    expression for each of those five types. For example, the excitation
    operator term in Hamiltonian when represented in terms of edge operators
    becomes:
        a_i^{\\dagger}a_j+a_j^{\\dagger}a_i = (-1j/2)*(A_ij*B_i+B_j*A_ij)
    For the sake of brevity the reader is encouraged to look up the
    expressions of other terms from the code below. The variables for edge
    operators are chosen according to the nomenclature defined above
    (B_i and A_ij). A detailed description of these operators and the terms
    of the electronic Hamiltonian are provided in (arXiv 1712.00446).

    Args:
        fer_op (FermionicOperator): the fermionic operator in the second quantized form

    Returns:
        WeightedPauliOperator: mapped qubit operator
    """
    fer_op = copy.deepcopy(fer_op)
    # bksf mapping works with the 'physicist' notation.
    fer_op.h2 = np.einsum('ijkm->ikmj', fer_op.h2)
    modes = fer_op.modes
    # Initialize qubit operator as constant.
    qubit_op = WeightedPauliOperator(paulis=[])
    edge_list = bravyi_kitaev_fast_edge_list(fer_op)
    # Loop through all indices.
    for p in range(modes):  # pylint: disable=invalid-name
        for q in range(modes):
            # Handle one-body terms.
            h1_pq = fer_op.h1[p, q]

            if h1_pq != 0.0 and p >= q:
                qubit_op += _one_body(edge_list, p, q, h1_pq)

            # Keep looping for the two-body terms.
            for r in range(modes):
                for s in range(modes):  # pylint: disable=invalid-name
                    h2_pqrs = fer_op.h2[p, q, r, s]

                    # Skip zero terms.
                    if (h2_pqrs == 0.0) or (p == q) or (r == s):
                        continue

                    # Identify and skip one of the complex conjugates.
                    if [p, q, r, s] != [s, r, q, p]:
                        if len(set([p, q, r, s])) == 4:
                            if min(r, s) < min(p, q):
                                continue
                        # Handle case of 3 unique indices
                        elif len(set([p, q, r, s])) == 3:
                            qubit_op += _two_body(edge_list, p, q, r, s, 0.5 * h2_pqrs)
                            continue
                        elif p != r and q < p:
                            continue

                    qubit_op += _two_body(edge_list, p, q, r, s, h2_pqrs)

    qubit_op.simplify()
    return qubit_op


def vacuum_operator(fer_op):
    """Use the stabilizers to find the vacuum state in bravyi_kitaev_fast.

    Args:
        fer_op (FermionicOperator): the fermionic operator in the second quantized form

    Returns:
        WeightedPauliOperator: the qubit operator
    """
    edge_list = bravyi_kitaev_fast_edge_list(fer_op)
    num_qubits = edge_list.shape[1]
    vac_operator = WeightedPauliOperator(paulis=[[1.0, Pauli.from_label('I' * num_qubits)]])

    graph = retworkx.PyGraph()
    graph.extend_from_edge_list(list(map(tuple, edge_list.transpose())))
    stabs = np.asarray(retworkx.cycle_basis(graph))
    for stab in stabs:
        a_op = WeightedPauliOperator(paulis=[[1.0, Pauli.from_label('I' * num_qubits)]])
        stab = np.asarray(stab)
        for i in range(np.size(stab)):
            a_op = a_op * edge_operator_aij(edge_list, stab[i], stab[(i + 1) % np.size(stab)]) * 1j
            # a_op.scaling_coeff(1j)
        a_op += WeightedPauliOperator(paulis=[[1.0, Pauli.from_label('I' * num_qubits)]])
        vac_operator = vac_operator * a_op * np.sqrt(2)
        # vac_operator.scaling_coeff()

    return vac_operator


def number_operator(fer_op, mode_number=None):
    """Find the qubit operator for the number operator in bravyi_kitaev_fast representation.

    Args:
        fer_op (FermionicOperator): the fermionic operator in the second quantized form
        mode_number (int): index, it corresponds to the mode for which number operator is required.

    Returns:
        WeightedPauliOperator: the qubit operator
    """
    modes = fer_op.h1.modes
    edge_list = bravyi_kitaev_fast_edge_list(fer_op)
    num_qubits = edge_list.shape[1]
    num_operator = WeightedPauliOperator(paulis=[[1.0, Pauli.from_label('I' * num_qubits)]])

    if mode_number is None:
        for i in range(modes):
            num_operator -= edge_operator_bi(edge_list, i)
        num_operator += \
            WeightedPauliOperator(paulis=[[1.0 * modes, Pauli.from_label('I' * num_qubits)]])
    else:
        num_operator += (WeightedPauliOperator(paulis=[[1.0, Pauli.from_label('I' * num_qubits)]])
                         - edge_operator_bi(edge_list, mode_number))

    num_operator = 0.5 * num_operator
    return num_operator


def generate_fermions(fer_op, i, j):
    """The qubit operator for generating fermions in bravyi_kitaev_fast representation.

    Args:
        fer_op (FermionicOperator): the fermionic operator in the second quantized form
        i (int): index of fermions
        j (int): index of fermions

    Returns:
        WeightedPauliOperator: the qubit operator
    """
    edge_list = bravyi_kitaev_fast_edge_list(fer_op)
    gen_fer_operator = edge_operator_aij(edge_list, i, j) * edge_operator_bi(edge_list, j) \
        - edge_operator_bi(edge_list, i) * edge_operator_aij(edge_list, i, j)

    gen_fer_operator = -0.5j * gen_fer_operator
    return gen_fer_operator
