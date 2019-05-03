# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import numpy as np


def sort(seq):
    """
    Tool function for normal order, should not be used separately

    Args:
        seq (list): array

    Returns:
        list: integer e.g. swapped array, number of swaps
    """
    swap_counter = 0
    changed = True
    while changed:
        changed = False
        for i in range(len(seq) - 1):
            if seq[i] > seq[i + 1]:
                swap_counter += 1
                seq[i], seq[i + 1] = seq[i + 1], seq[i]
                changed = True

    return seq, swap_counter


def last_two_indices_swap(array_ind_two_body_term):
    """
    Swap 2 last indices of an array

    Args:
        array_ind_two_body_term (list): TBD

    Returns:
        list: TBD
    """
    swapped_indices = [0, 0, 0, 0]
    swapped_indices[0] = array_ind_two_body_term[0]
    swapped_indices[1] = array_ind_two_body_term[1]
    swapped_indices[2] = array_ind_two_body_term[3]
    swapped_indices[3] = array_ind_two_body_term[2]

    return swapped_indices


def normal_order_integrals(n_qubits, n_occupied, array_to_normal_order, array_mapping, h1_old, h2_old,
                           h1_new, h2_new):
    """
    Given an operator and the rFs and rsgtu from Gaussian it produces new
    h1,h2,id_terms usable for the generation of the Hamiltonian in Pauli strings form.

    Args:
        n_qubits (int): number of qubits
        n_occupied (int): number of electrons (occupied orbitals)
        array_to_normal_order (list):  e.g. [i,j,k,l] indices of the term to normal order
        array_mapping (list): e.g. two body terms list  ['adag', 'adag', 'a', 'a'],
        single body terms list (list): ['adag', 'a']
        h1_old (numpy.ndarray): e.g. rFs.dat (dim(rsgtu) = [n_qubits,n_qubits,n_qubits,n_qubits])
                       loaded with QuTip function (qutip.fileio.qload) or numpy.array
        h2_old (numpy.ndarray): e.g. rsgtu.dat (dim(rsgtu) = [n_qubits,n_qubits])
        h1_new (numpy.ndarray): e.g. numpy.zeros([n_qubits, n_qubits])
        h2_new (numpy.ndarray): e.g. numpy.zeros([n_qubits, n_qubits, n_qubits, n_qubits])

    Returns:
        numpy.ndarray, numpy.ndarray, float: h1_new, h2_new, id_term
    """
    a_enum = []
    adag_enum = []

    for ind in range(n_qubits):
        if ind < n_occupied:
            a_enum.append(-(ind + 1))
            adag_enum.append(ind + 1)
        else:
            a_enum.append(ind + 1)
            adag_enum.append(-(ind + 1))

    array_to_sort = []

    for ind in range(len(array_to_normal_order)):
        if array_mapping[ind] == "adag":
            array_to_sort.append(adag_enum[array_to_normal_order[ind]])
        elif array_mapping[ind] == "a":
            array_to_sort.append(a_enum[array_to_normal_order[ind]])

    sign = (-1.) ** sort(array_to_sort)[1]
    array_sorted = sort(array_to_sort)[0]

    ind_ini_term = array_to_normal_order

    mapping_no_term = []
    ind_no_term = []
    sign_no_term = sign

    for ind in array_sorted:
        if ind in a_enum:
            mapping_no_term.append("a")
            ind_no_term.append(a_enum.index(ind))
        elif ind in adag_enum:
            mapping_no_term.append("adag")
            ind_no_term.append(adag_enum.index(ind))

    ii = 0
    jj = 1
    kk = 2
    ll = 3

    id_term = 0.

    if len(array_to_normal_order) == 2:
        if ind_no_term[0] == ind_no_term[1]:
            if mapping_no_term == ['adag', 'a']:
                temp_sign_h1 = float(1 * sign_no_term)

                ind_old_h1 = [ind_ini_term[ii], ind_ini_term[jj]]
                ind_new_h1 = [ind_no_term[ii], ind_no_term[jj]]

                h1_new[ind_new_h1[0]][ind_new_h1[1]] \
                    += float(
                    temp_sign_h1 * h1_old[ind_old_h1[0]][ind_old_h1[1]])

            elif mapping_no_term == ['a', 'adag']:
                temp_sign_h1 = float(-1 * sign_no_term)

                ind_old_h1 = [ind_ini_term[ii], ind_ini_term[jj]]
                ind_new_h1 = [ind_no_term[jj], ind_no_term[ii]]

                h1_new[ind_new_h1[0]][ind_new_h1[1]] \
                    += float(
                    temp_sign_h1 * h1_old[ind_old_h1[0]][ind_old_h1[1]])

                id_term += float(
                    sign_no_term * h1_old[ind_old_h1[0]][
                        ind_old_h1[1]])

        else:
            if mapping_no_term == ['adag', 'a']:
                temp_sign_h1 = float(1 * sign_no_term)

                ind_old_h1 = [ind_ini_term[ii], ind_ini_term[jj]]
                ind_new_h1 = [ind_no_term[ii], ind_no_term[jj]]

                h1_new[ind_new_h1[0]][ind_new_h1[1]] \
                    += float(
                    temp_sign_h1 * h1_old[ind_old_h1[0]][ind_old_h1[1]])

            elif mapping_no_term == ['a', 'adag']:
                temp_sign_h1 = float(-1 * sign_no_term)

                ind_old_h1 = [ind_ini_term[ii], ind_ini_term[jj]]
                ind_new_h1 = [ind_no_term[jj], ind_no_term[ii]]

                h1_new[ind_new_h1[0]][ind_new_h1[1]] \
                    += float(
                    temp_sign_h1 * h1_old[ind_old_h1[0]][ind_old_h1[1]])

    elif len(array_to_normal_order) == 4:
        if len(set(ind_no_term)) == 4:
            if mapping_no_term == ['adag', 'adag', 'a', 'a']:
                temp_sign_h2 = 1 * sign_no_term

                ind_new_h2 = [ind_no_term[ii], ind_no_term[jj],
                              ind_no_term[kk], ind_no_term[ll]]
                ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                              ind_ini_term[2], ind_ini_term[3]]
                ind_old_h2 = last_two_indices_swap(ind_old_h2)

                h2_new[ind_new_h2[0]][ind_new_h2[1]][
                    ind_new_h2[2]][ind_new_h2[3]] \
                    += 0.5 * temp_sign_h2 * \
                    h2_old[ind_old_h2[0]][ind_old_h2[1]][
                    ind_old_h2[2]][ind_old_h2[3]]

            elif mapping_no_term == ['adag', 'a', 'adag', 'a']:
                temp_sign_h2 = -1 * sign_no_term

                ind_new_h2 = [ind_no_term[ii], ind_no_term[kk],
                              ind_no_term[jj], ind_no_term[ll]]
                ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                              ind_ini_term[2], ind_ini_term[3]]
                ind_old_h2 = last_two_indices_swap(ind_old_h2)

                h2_new[ind_new_h2[0]][ind_new_h2[1]][
                    ind_new_h2[2]][ind_new_h2[3]] \
                    += 0.5 * temp_sign_h2 * \
                    h2_old[ind_old_h2[0]][ind_old_h2[1]][
                    ind_old_h2[2]][ind_old_h2[3]]

            elif mapping_no_term == ['adag', 'a', 'a', 'adag']:
                temp_sign_h2 = 1 * sign_no_term

                ind_new_h2 = [ind_no_term[ii], ind_no_term[ll],
                              ind_no_term[jj], ind_no_term[kk]]
                ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                              ind_ini_term[2], ind_ini_term[3]]
                ind_old_h2 = last_two_indices_swap(ind_old_h2)

                h2_new[ind_new_h2[0]][ind_new_h2[1]][
                    ind_new_h2[2]][ind_new_h2[3]] \
                    += 0.5 * temp_sign_h2 * \
                    h2_old[ind_old_h2[0]][ind_old_h2[1]][
                    ind_old_h2[2]][ind_old_h2[3]]

            elif mapping_no_term == ['a', 'adag', 'adag', 'a']:
                temp_sign_h2 = 1 * sign_no_term

                ind_new_h2 = [ind_no_term[jj], ind_no_term[kk],
                              ind_no_term[ii], ind_no_term[ll]]
                ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                              ind_ini_term[2], ind_ini_term[3]]
                ind_old_h2 = last_two_indices_swap(ind_old_h2)

                h2_new[ind_new_h2[0]][ind_new_h2[1]][
                    ind_new_h2[2]][ind_new_h2[3]] \
                    += 0.5 * temp_sign_h2 * \
                    h2_old[ind_old_h2[0]][ind_old_h2[1]][
                    ind_old_h2[2]][ind_old_h2[3]]

            elif mapping_no_term == ['a', 'adag', 'a', 'adag']:
                temp_sign_h2 = -1 * sign_no_term

                ind_new_h2 = [ind_no_term[jj], ind_no_term[ll],
                              ind_no_term[ii], ind_no_term[kk]]
                ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                              ind_ini_term[2], ind_ini_term[3]]
                ind_old_h2 = last_two_indices_swap(ind_old_h2)

                h2_new[ind_new_h2[0]][ind_new_h2[1]][
                    ind_new_h2[2]][ind_new_h2[3]] \
                    += 0.5 * temp_sign_h2 * \
                    h2_old[ind_old_h2[0]][ind_old_h2[1]][
                    ind_old_h2[2]][ind_old_h2[3]]

            elif mapping_no_term == ['a', 'a', 'adag', 'adag']:
                temp_sign_h2 = 1 * sign_no_term

                ind_new_h2 = [ind_no_term[kk], ind_no_term[ll],
                              ind_no_term[ii], ind_no_term[jj]]
                ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                              ind_ini_term[2], ind_ini_term[3]]
                ind_old_h2 = last_two_indices_swap(ind_old_h2)

                h2_new[ind_new_h2[0]][ind_new_h2[1]][
                    ind_new_h2[2]][ind_new_h2[3]] \
                    += 0.5 * temp_sign_h2 * \
                    h2_old[ind_old_h2[0]][ind_old_h2[1]][
                    ind_old_h2[2]][ind_old_h2[3]]

            else:
                print('ERROR 1')

        elif len(set(ind_no_term)) == 3:

            if ind_no_term[0] == ind_no_term[1]:

                if mapping_no_term == ['adag', 'adag', 'a', 'a']:

                    temp_sign_h2 = 1 * sign_no_term

                    ind_new_h2 = [ind_no_term[ii],
                                  ind_no_term[ii],
                                  ind_no_term[kk],
                                  ind_no_term[ll]]
                    ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                  ind_ini_term[2], ind_ini_term[3]]
                    ind_old_h2 = last_two_indices_swap(ind_old_h2)

                    h2_new[ind_new_h2[0]][ind_new_h2[1]][
                        ind_new_h2[2]][ind_new_h2[3]] \
                        += 0.5 * temp_sign_h2 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                elif mapping_no_term == ['adag', 'a', 'adag', 'a']:

                    temp_sign_h2 = -1 * sign_no_term

                    ind_new_h2 = [ind_no_term[ii],
                                  ind_no_term[kk],
                                  ind_no_term[ii],
                                  ind_no_term[ll]]
                    ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                  ind_ini_term[2], ind_ini_term[3]]
                    ind_old_h2 = last_two_indices_swap(ind_old_h2)

                    h2_new[ind_new_h2[0]][ind_new_h2[1]][
                        ind_new_h2[2]][ind_new_h2[3]] \
                        += 0.5 * temp_sign_h2 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                elif mapping_no_term == ['adag', 'a', 'a', 'adag']:

                    temp_sign_h2 = 1 * sign_no_term

                    ind_new_h2 = [ind_no_term[ii],
                                  ind_no_term[ll],
                                  ind_no_term[ii],
                                  ind_no_term[kk]]
                    ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                  ind_ini_term[2], ind_ini_term[3]]
                    ind_old_h2 = last_two_indices_swap(ind_old_h2)

                    h2_new[ind_new_h2[0]][ind_new_h2[1]][
                        ind_new_h2[2]][ind_new_h2[3]] \
                        += 0.5 * temp_sign_h2 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                elif mapping_no_term == ['a', 'adag', 'adag', 'a']:

                    temp_sign_h2 = 1 * sign_no_term
                    temp_sign_h1 = 1 * sign_no_term

                    ind_new_h2 = [ind_no_term[ii],
                                  ind_no_term[kk],
                                  ind_no_term[ii],
                                  ind_no_term[ll]]
                    ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                  ind_ini_term[2], ind_ini_term[3]]
                    ind_old_h2 = last_two_indices_swap(ind_old_h2)

                    ind_old_h1 = [ind_no_term[kk],
                                  ind_no_term[ll]]

                    h2_new[ind_new_h2[0]][ind_new_h2[1]][
                        ind_new_h2[2]][ind_new_h2[3]] \
                        += 0.5 * temp_sign_h2 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                    h1_new[ind_old_h1[0]][ind_old_h1[1]] \
                        += 0.5 * temp_sign_h1 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                elif mapping_no_term == ['a', 'adag', 'a', 'adag']:

                    temp_sign_h2 = -1 * sign_no_term
                    temp_sign_h1 = -1 * sign_no_term

                    ind_new_h2 = [ind_no_term[ii],
                                  ind_no_term[ll],
                                  ind_no_term[ii],
                                  ind_no_term[kk]]
                    ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                  ind_ini_term[2], ind_ini_term[3]]
                    ind_old_h2 = last_two_indices_swap(ind_old_h2)

                    ind_old_h1 = [ind_no_term[ll],
                                  ind_no_term[kk]]

                    h2_new[ind_new_h2[0]][ind_new_h2[1]][
                        ind_new_h2[2]][ind_new_h2[3]] \
                        += 0.5 * temp_sign_h2 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                    h1_new[ind_old_h1[0]][ind_old_h1[1]] \
                        += 0.5 * temp_sign_h1 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                elif mapping_no_term == ['a', 'a', 'adag', 'adag']:

                    temp_sign_h2 = 1 * sign_no_term

                    ind_new_h2 = [ind_no_term[kk],
                                  ind_no_term[ll],
                                  ind_no_term[ii],
                                  ind_no_term[ii]]
                    ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                  ind_ini_term[2], ind_ini_term[3]]
                    ind_old_h2 = last_two_indices_swap(ind_old_h2)

                    h2_new[ind_new_h2[0]][ind_new_h2[1]][
                        ind_new_h2[2]][ind_new_h2[3]] \
                        += 0.5 * temp_sign_h2 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                else:
                    print('ERROR 2')

            elif ind_no_term[0] == ind_no_term[2]:

                if mapping_no_term == ['adag', 'adag', 'a', 'a']:

                    temp_sign_h2 = 1 * sign_no_term

                    ind_new_h2 = [ind_no_term[ii],
                                  ind_no_term[jj],
                                  ind_no_term[ii],
                                  ind_no_term[ll]]
                    ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                  ind_ini_term[2], ind_ini_term[3]]
                    ind_old_h2 = last_two_indices_swap(ind_old_h2)

                    h2_new[ind_new_h2[0]][ind_new_h2[1]][
                        ind_new_h2[2]][ind_new_h2[3]] \
                        += 0.5 * temp_sign_h2 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                elif mapping_no_term == ['adag', 'a', 'adag', 'a']:

                    temp_sign_h2 = -1 * sign_no_term

                    ind_new_h2 = [ind_no_term[ii],
                                  ind_no_term[ii],
                                  ind_no_term[jj],
                                  ind_no_term[ll]]
                    ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                  ind_ini_term[2], ind_ini_term[3]]
                    ind_old_h2 = last_two_indices_swap(ind_old_h2)

                    h2_new[ind_new_h2[0]][ind_new_h2[1]][
                        ind_new_h2[2]][ind_new_h2[3]] \
                        += 0.5 * temp_sign_h2 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                elif mapping_no_term == ['adag', 'a', 'a', 'adag']:

                    temp_sign_h2 = 1 * sign_no_term

                    ind_new_h2 = [ind_no_term[ii],
                                  ind_no_term[ll],
                                  ind_no_term[jj],
                                  ind_no_term[ii]]
                    ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                  ind_ini_term[2], ind_ini_term[3]]
                    ind_old_h2 = last_two_indices_swap(ind_old_h2)

                    h2_new[ind_new_h2[0]][ind_new_h2[1]][
                        ind_new_h2[2]][ind_new_h2[3]] \
                        += 0.5 * temp_sign_h2 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                elif mapping_no_term == ['a', 'adag', 'adag', 'a']:

                    temp_sign_h2 = 1 * sign_no_term
                    temp_sign_h1 = -1 * sign_no_term

                    ind_new_h2 = [ind_no_term[jj],
                                  ind_no_term[ii],
                                  ind_no_term[ii],
                                  ind_no_term[ll]]
                    ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                  ind_ini_term[2], ind_ini_term[3]]
                    ind_old_h2 = last_two_indices_swap(ind_old_h2)

                    ind_old_h1 = [ind_no_term[jj],
                                  ind_no_term[ll]]

                    h2_new[ind_new_h2[0]][ind_new_h2[1]][
                        ind_new_h2[2]][ind_new_h2[3]] \
                        += 0.5 * temp_sign_h2 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                    h1_new[ind_old_h1[0]][ind_old_h1[1]] \
                        += 0.5 * temp_sign_h1 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                elif mapping_no_term == ['a', 'adag', 'a', 'adag']:

                    temp_sign_h2 = -1 * sign_no_term

                    ind_new_h2 = [ind_no_term[jj],
                                  ind_no_term[ll],
                                  ind_no_term[ii],
                                  ind_no_term[ii]]
                    ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                  ind_ini_term[2], ind_ini_term[3]]
                    ind_old_h2 = last_two_indices_swap(ind_old_h2)

                    h2_new[ind_new_h2[0]][ind_new_h2[1]][
                        ind_new_h2[2]][ind_new_h2[3]] \
                        += 0.5 * temp_sign_h2 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                elif mapping_no_term == ['a', 'a', 'adag', 'adag']:

                    temp_sign_h2 = 1 * sign_no_term
                    temp_sign_h1 = 1 * sign_no_term

                    ind_new_h2 = [ind_no_term[ii],
                                  ind_no_term[ll],
                                  ind_no_term[ii],
                                  ind_no_term[jj]]
                    ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                  ind_ini_term[2], ind_ini_term[3]]
                    ind_old_h2 = last_two_indices_swap(ind_old_h2)

                    ind_old_h1 = [ind_no_term[ll],
                                  ind_no_term[jj]]

                    h2_new[ind_new_h2[0]][ind_new_h2[1]][
                        ind_new_h2[2]][ind_new_h2[3]] \
                        += 0.5 * temp_sign_h2 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                    h1_new[ind_old_h1[0]][ind_old_h1[1]] \
                        += 0.5 * temp_sign_h1 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                else:
                    print('ERROR 3')

            elif ind_no_term[0] == ind_no_term[3]:

                if mapping_no_term == ['adag', 'adag', 'a', 'a']:

                    temp_sign_h2 = 1 * sign_no_term

                    ind_new_h2 = [ind_no_term[ii],
                                  ind_no_term[jj],
                                  ind_no_term[kk],
                                  ind_no_term[ii]]
                    ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                  ind_ini_term[2], ind_ini_term[3]]
                    ind_old_h2 = last_two_indices_swap(ind_old_h2)

                    h2_new[ind_new_h2[0]][ind_new_h2[1]][
                        ind_new_h2[2]][ind_new_h2[3]] \
                        += 0.5 * temp_sign_h2 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                elif mapping_no_term == ['adag', 'a', 'adag', 'a']:

                    temp_sign_h2 = -1 * sign_no_term

                    ind_new_h2 = [ind_no_term[ii],
                                  ind_no_term[kk],
                                  ind_no_term[jj],
                                  ind_no_term[ii]]
                    ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                  ind_ini_term[2], ind_ini_term[3]]
                    ind_old_h2 = last_two_indices_swap(ind_old_h2)

                    h2_new[ind_new_h2[0]][ind_new_h2[1]][
                        ind_new_h2[2]][ind_new_h2[3]] \
                        += 0.5 * temp_sign_h2 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                elif mapping_no_term == ['adag', 'a', 'a', 'adag']:

                    temp_sign_h2 = 1 * sign_no_term

                    ind_new_h2 = [ind_no_term[ii],
                                  ind_no_term[ii],
                                  ind_no_term[jj],
                                  ind_no_term[kk]]
                    ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                  ind_ini_term[2], ind_ini_term[3]]
                    ind_old_h2 = last_two_indices_swap(ind_old_h2)

                    h2_new[ind_new_h2[0]][ind_new_h2[1]][
                        ind_new_h2[2]][ind_new_h2[3]] \
                        += 0.5 * temp_sign_h2 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                elif mapping_no_term == ['a', 'adag', 'adag', 'a']:

                    temp_sign_h2 = 1 * sign_no_term

                    ind_new_h2 = [ind_no_term[jj],
                                  ind_no_term[kk],
                                  ind_no_term[ii],
                                  ind_no_term[ii]]
                    ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                  ind_ini_term[2], ind_ini_term[3]]
                    ind_old_h2 = last_two_indices_swap(ind_old_h2)

                    h2_new[ind_new_h2[0]][ind_new_h2[1]][
                        ind_new_h2[2]][ind_new_h2[3]] \
                        += 0.5 * temp_sign_h2 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                elif mapping_no_term == ['a', 'adag', 'a', 'adag']:

                    temp_sign_h2 = -1 * sign_no_term
                    temp_sign_h1 = 1 * sign_no_term

                    ind_new_h2 = [ind_no_term[jj],
                                  ind_no_term[ii],
                                  ind_no_term[ii],
                                  ind_no_term[kk]]
                    ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                  ind_ini_term[2], ind_ini_term[3]]
                    ind_old_h2 = last_two_indices_swap(ind_old_h2)

                    ind_old_h1 = [ind_no_term[jj],
                                  ind_no_term[kk]]

                    h2_new[ind_new_h2[0]][ind_new_h2[1]][
                        ind_new_h2[2]][ind_new_h2[3]] \
                        += 0.5 * temp_sign_h2 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                    h1_new[ind_old_h1[0]][ind_old_h1[1]] \
                        += 0.5 * temp_sign_h1 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                elif mapping_no_term == ['a', 'a', 'adag', 'adag']:

                    temp_sign_h2 = 1 * sign_no_term
                    temp_sign_h1 = -1 * sign_no_term

                    ind_new_h2 = [ind_no_term[kk],
                                  ind_no_term[ii],
                                  ind_no_term[ii],
                                  ind_no_term[jj]]
                    ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                  ind_ini_term[2], ind_ini_term[3]]
                    ind_old_h2 = last_two_indices_swap(ind_old_h2)

                    ind_old_h1 = [ind_no_term[kk],
                                  ind_no_term[jj]]

                    h2_new[ind_new_h2[0]][ind_new_h2[1]][
                        ind_new_h2[2]][ind_new_h2[3]] \
                        += 0.5 * temp_sign_h2 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                    h1_new[ind_old_h1[0]][ind_old_h1[1]] \
                        += 0.5 * temp_sign_h1 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]
                else:
                    print('ERROR 4')

            elif ind_no_term[1] == ind_no_term[2]:

                if mapping_no_term == ['adag', 'adag', 'a', 'a']:

                    temp_sign_h2 = 1 * sign_no_term

                    ind_new_h2 = [ind_no_term[ii],
                                  ind_no_term[jj],
                                  ind_no_term[jj],
                                  ind_no_term[ll]]
                    ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                  ind_ini_term[2], ind_ini_term[3]]
                    ind_old_h2 = last_two_indices_swap(ind_old_h2)

                    h2_new[ind_new_h2[0]][ind_new_h2[1]][
                        ind_new_h2[2]][ind_new_h2[3]] \
                        += 0.5 * temp_sign_h2 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                elif mapping_no_term == ['adag', 'a', 'adag', 'a']:

                    temp_sign_h2 = -1 * sign_no_term
                    temp_sign_h1 = 1 * sign_no_term

                    ind_new_h2 = [ind_no_term[ii],
                                  ind_no_term[jj],
                                  ind_no_term[jj],
                                  ind_no_term[ll]]
                    ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                  ind_ini_term[2], ind_ini_term[3]]
                    ind_old_h2 = last_two_indices_swap(ind_old_h2)

                    ind_old_h1 = [ind_no_term[ii],
                                  ind_no_term[ll]]

                    h2_new[ind_new_h2[0]][ind_new_h2[1]][
                        ind_new_h2[2]][ind_new_h2[3]] \
                        += 0.5 * temp_sign_h2 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                    h1_new[ind_old_h1[0]][ind_old_h1[1]] \
                        += 0.5 * temp_sign_h1 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                elif mapping_no_term == ['adag', 'a', 'a', 'adag']:

                    temp_sign_h2 = 1 * sign_no_term

                    ind_new_h2 = [ind_no_term[ii],
                                  ind_no_term[ll],
                                  ind_no_term[jj],
                                  ind_no_term[jj]]
                    ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                  ind_ini_term[2], ind_ini_term[3]]
                    ind_old_h2 = last_two_indices_swap(ind_old_h2)

                    h2_new[ind_new_h2[0]][ind_new_h2[1]][
                        ind_new_h2[2]][ind_new_h2[3]] \
                        += 0.5 * temp_sign_h2 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                elif mapping_no_term == ['a', 'adag', 'adag', 'a']:

                    temp_sign_h2 = 1 * sign_no_term

                    ind_new_h2 = [ind_no_term[jj],
                                  ind_no_term[jj],
                                  ind_no_term[ii],
                                  ind_no_term[ll]]
                    ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                  ind_ini_term[2], ind_ini_term[3]]
                    ind_old_h2 = last_two_indices_swap(ind_old_h2)

                    h2_new[ind_new_h2[0]][ind_new_h2[1]][
                        ind_new_h2[2]][ind_new_h2[3]] \
                        += 0.5 * temp_sign_h2 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                elif mapping_no_term == ['a', 'adag', 'a', 'adag']:

                    temp_sign_h2 = -1 * sign_no_term

                    ind_new_h2 = [ind_no_term[jj],
                                  ind_no_term[ll],
                                  ind_no_term[ii],
                                  ind_no_term[jj]]
                    ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                  ind_ini_term[2], ind_ini_term[3]]
                    ind_old_h2 = last_two_indices_swap(ind_old_h2)

                    h2_new[ind_new_h2[0]][ind_new_h2[1]][
                        ind_new_h2[2]][ind_new_h2[3]] \
                        += 0.5 * temp_sign_h2 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                elif mapping_no_term == ['a', 'a', 'adag', 'adag']:

                    temp_sign_h2 = 1 * sign_no_term
                    temp_sign_h1 = -1 * sign_no_term

                    ind_new_h2 = [ind_no_term[jj],
                                  ind_no_term[ll],
                                  ind_no_term[ii],
                                  ind_no_term[jj]]
                    ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                  ind_ini_term[2], ind_ini_term[3]]
                    ind_old_h2 = last_two_indices_swap(ind_old_h2)

                    ind_old_h1 = [ind_no_term[ll],
                                  ind_no_term[ii]]

                    h2_new[ind_new_h2[0]][ind_new_h2[1]][
                        ind_new_h2[2]][ind_new_h2[3]] \
                        += 0.5 * temp_sign_h2 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                    h1_new[ind_old_h1[0]][ind_old_h1[1]] \
                        += 0.5 * temp_sign_h1 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                else:
                    print('ERROR 5')

            elif ind_no_term[1] == ind_no_term[3]:

                if mapping_no_term == ['adag', 'adag', 'a', 'a']:

                    temp_sign_h2 = 1 * sign_no_term

                    ind_new_h2 = [ind_no_term[ii],
                                  ind_no_term[jj],
                                  ind_no_term[kk],
                                  ind_no_term[jj]]
                    ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                  ind_ini_term[2], ind_ini_term[3]]
                    ind_old_h2 = last_two_indices_swap(ind_old_h2)

                    h2_new[ind_new_h2[0]][ind_new_h2[1]][
                        ind_new_h2[2]][ind_new_h2[3]] \
                        += 0.5 * temp_sign_h2 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                elif mapping_no_term == ['adag', 'a', 'adag', 'a']:

                    temp_sign_h2 = -1 * sign_no_term

                    ind_new_h2 = [ind_no_term[ii],
                                  ind_no_term[kk],
                                  ind_no_term[jj],
                                  ind_no_term[jj]]
                    ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                  ind_ini_term[2], ind_ini_term[3]]
                    ind_old_h2 = last_two_indices_swap(ind_old_h2)

                    h2_new[ind_new_h2[0]][ind_new_h2[1]][
                        ind_new_h2[2]][ind_new_h2[3]] \
                        += 0.5 * temp_sign_h2 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                elif mapping_no_term == ['adag', 'a', 'a', 'adag']:

                    temp_sign_h2 = 1 * sign_no_term
                    temp_sign_h1 = -1 * sign_no_term

                    ind_new_h2 = [ind_no_term[ii],
                                  ind_no_term[jj],
                                  ind_no_term[jj],
                                  ind_no_term[kk]]
                    ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                  ind_ini_term[2], ind_ini_term[3]]
                    ind_old_h2 = last_two_indices_swap(ind_old_h2)

                    ind_old_h1 = [ind_no_term[ii],
                                  ind_no_term[kk]]

                    h2_new[ind_new_h2[0]][ind_new_h2[1]][
                        ind_new_h2[2]][ind_new_h2[3]] \
                        += 0.5 * temp_sign_h2 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                    h1_new[ind_old_h1[0]][ind_old_h1[1]] \
                        += 0.5 * temp_sign_h1 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                elif mapping_no_term == ['a', 'adag', 'adag', 'a']:

                    temp_sign_h2 = 1 * sign_no_term

                    ind_new_h2 = [ind_no_term[jj],
                                  ind_no_term[kk],
                                  ind_no_term[ii],
                                  ind_no_term[jj]]
                    ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                  ind_ini_term[2], ind_ini_term[3]]
                    ind_old_h2 = last_two_indices_swap(ind_old_h2)

                    h2_new[ind_new_h2[0]][ind_new_h2[1]][
                        ind_new_h2[2]][ind_new_h2[3]] \
                        += 0.5 * temp_sign_h2 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                elif mapping_no_term == ['a', 'adag', 'a', 'adag']:

                    temp_sign_h2 = -1 * sign_no_term

                    ind_new_h2 = [ind_no_term[jj],
                                  ind_no_term[jj],
                                  ind_no_term[ii],
                                  ind_no_term[kk]]
                    ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                  ind_ini_term[2], ind_ini_term[3]]
                    ind_old_h2 = last_two_indices_swap(ind_old_h2)

                    h2_new[ind_new_h2[0]][ind_new_h2[1]][
                        ind_new_h2[2]][ind_new_h2[3]] \
                        += 0.5 * temp_sign_h2 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                elif mapping_no_term == ['a', 'a', 'adag', 'adag']:

                    temp_sign_h2 = 1 * sign_no_term
                    temp_sign_h1 = 1 * sign_no_term

                    ind_new_h2 = [ind_no_term[kk],
                                  ind_no_term[jj],
                                  ind_no_term[ii],
                                  ind_no_term[jj]]
                    ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                  ind_ini_term[2], ind_ini_term[3]]
                    ind_old_h2 = last_two_indices_swap(ind_old_h2)

                    ind_old_h1 = [ind_no_term[kk],
                                  ind_no_term[ii]]

                    h2_new[ind_new_h2[0]][ind_new_h2[1]][
                        ind_new_h2[2]][ind_new_h2[3]] \
                        += 0.5 * temp_sign_h2 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                    h1_new[ind_old_h1[0]][ind_old_h1[1]] \
                        += 0.5 * temp_sign_h1 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                else:
                    print('ERROR 6')

            elif ind_no_term[2] == ind_no_term[3]:

                if mapping_no_term == ['adag', 'adag', 'a', 'a']:

                    temp_sign_h2 = 1 * sign_no_term

                    ind_new_h2 = [ind_no_term[ii],
                                  ind_no_term[jj],
                                  ind_no_term[kk],
                                  ind_no_term[kk]]
                    ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                  ind_ini_term[2], ind_ini_term[3]]
                    ind_old_h2 = last_two_indices_swap(ind_old_h2)

                    h2_new[ind_new_h2[0]][ind_new_h2[1]][
                        ind_new_h2[2]][ind_new_h2[3]] \
                        += 0.5 * temp_sign_h2 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                elif mapping_no_term == ['adag', 'a', 'adag', 'a']:

                    temp_sign_h2 = -1 * sign_no_term

                    ind_new_h2 = [ind_no_term[ii],
                                  ind_no_term[kk],
                                  ind_no_term[jj],
                                  ind_no_term[kk]]
                    ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                  ind_ini_term[2], ind_ini_term[3]]
                    ind_old_h2 = last_two_indices_swap(ind_old_h2)

                    h2_new[ind_new_h2[0]][ind_new_h2[1]][
                        ind_new_h2[2]][ind_new_h2[3]] \
                        += 0.5 * temp_sign_h2 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                elif mapping_no_term == ['adag', 'a', 'a', 'adag']:

                    temp_sign_h2 = 1 * sign_no_term
                    temp_sign_h1 = 1 * sign_no_term

                    ind_new_h2 = [ind_no_term[ii],
                                  ind_no_term[kk],
                                  ind_no_term[jj],
                                  ind_no_term[kk]]
                    ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                  ind_ini_term[2], ind_ini_term[3]]
                    ind_old_h2 = last_two_indices_swap(ind_old_h2)

                    ind_old_h1 = [ind_no_term[ii],
                                  ind_no_term[jj]]

                    h2_new[ind_new_h2[0]][ind_new_h2[1]][
                        ind_new_h2[2]][ind_new_h2[3]] \
                        += 0.5 * temp_sign_h2 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                    h1_new[ind_old_h1[0]][ind_old_h1[1]] \
                        += 0.5 * temp_sign_h1 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                elif mapping_no_term == ['a', 'adag', 'adag', 'a']:

                    temp_sign_h2 = 1 * sign_no_term

                    ind_new_h2 = [ind_no_term[jj],
                                  ind_no_term[kk],
                                  ind_no_term[ii],
                                  ind_no_term[kk]]
                    ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                  ind_ini_term[2], ind_ini_term[3]]
                    ind_old_h2 = last_two_indices_swap(ind_old_h2)

                    h2_new[ind_new_h2[0]][ind_new_h2[1]][
                        ind_new_h2[2]][ind_new_h2[3]] \
                        += 0.5 * temp_sign_h2 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                elif mapping_no_term == ['a', 'adag', 'a', 'adag']:

                    temp_sign_h2 = -1 * sign_no_term
                    temp_sign_h1 = -1 * sign_no_term

                    ind_new_h2 = [ind_no_term[jj],
                                  ind_no_term[kk],
                                  ind_no_term[ii],
                                  ind_no_term[kk]]
                    ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                  ind_ini_term[2], ind_ini_term[3]]
                    ind_old_h2 = last_two_indices_swap(ind_old_h2)

                    ind_old_h1 = [ind_no_term[jj],
                                  ind_no_term[ii]]

                    h2_new[ind_new_h2[0]][ind_new_h2[1]][
                        ind_new_h2[2]][ind_new_h2[3]] \
                        += 0.5 * temp_sign_h2 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                    h1_new[ind_old_h1[0]][ind_old_h1[1]] \
                        += 0.5 * temp_sign_h1 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                elif mapping_no_term == ['a', 'a', 'adag', 'adag']:

                    temp_sign_h2 = 1 * sign_no_term

                    ind_new_h2 = [ind_no_term[kk],
                                  ind_no_term[kk],
                                  ind_no_term[ii],
                                  ind_no_term[jj]]
                    ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                  ind_ini_term[2], ind_ini_term[3]]
                    ind_old_h2 = last_two_indices_swap(ind_old_h2)

                    h2_new[ind_new_h2[0]][ind_new_h2[1]][
                        ind_new_h2[2]][ind_new_h2[3]] \
                        += 0.5 * temp_sign_h2 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                else:
                    print('ERROR 7')

            else:
                print('ERROR 8')

        elif len(set(ind_no_term)) == 2:

            if ind_no_term[0] == ind_no_term[1] and \
                    ind_no_term[2] == ind_no_term[3]:

                if mapping_no_term == ['adag', 'adag', 'a', 'a']:

                    temp_sign_h2 = 1 * sign_no_term

                    ind_new_h2 = [ind_no_term[ii],
                                  ind_no_term[ii],
                                  ind_no_term[kk],
                                  ind_no_term[kk]]
                    ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                  ind_ini_term[2], ind_ini_term[3]]
                    ind_old_h2 = last_two_indices_swap(ind_old_h2)

                    h2_new[ind_new_h2[0]][ind_new_h2[1]][
                        ind_new_h2[2]][ind_new_h2[3]] \
                        += 0.5 * temp_sign_h2 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                elif mapping_no_term == ['adag', 'a', 'adag', 'a']:

                    temp_sign_h2 = -1 * sign_no_term

                    ind_new_h2 = [ind_no_term[ii],
                                  ind_no_term[kk],
                                  ind_no_term[ii],
                                  ind_no_term[kk]]
                    ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                  ind_ini_term[2], ind_ini_term[3]]
                    ind_old_h2 = last_two_indices_swap(ind_old_h2)

                    h2_new[ind_new_h2[0]][ind_new_h2[1]][
                        ind_new_h2[2]][ind_new_h2[3]] \
                        += 0.5 * temp_sign_h2 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                elif mapping_no_term == ['adag', 'a', 'a', 'adag']:

                    temp_sign_h2 = -1 * sign_no_term
                    temp_sign_h1 = 1 * sign_no_term

                    ind_new_h2 = [ind_no_term[ii],
                                  ind_no_term[kk],
                                  ind_no_term[kk],
                                  ind_no_term[ii]]
                    ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                  ind_ini_term[2], ind_ini_term[3]]
                    ind_old_h2 = last_two_indices_swap(ind_old_h2)

                    ind_old_h1 = [ind_no_term[ii],
                                  ind_no_term[ii]]

                    h2_new[ind_new_h2[0]][ind_new_h2[1]][
                        ind_new_h2[2]][ind_new_h2[3]] \
                        += 0.5 * temp_sign_h2 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                    h1_new[ind_old_h1[0]][ind_old_h1[1]] \
                        += 0.5 * temp_sign_h1 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                elif mapping_no_term == ['a', 'adag', 'adag', 'a']:

                    temp_sign_h2 = 1 * sign_no_term
                    temp_sign_h1 = 1 * sign_no_term

                    ind_new_h2 = [ind_no_term[ii],
                                  ind_no_term[kk],
                                  ind_no_term[ii],
                                  ind_no_term[kk]]
                    ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                  ind_ini_term[2], ind_ini_term[3]]
                    ind_old_h2 = last_two_indices_swap(ind_old_h2)

                    ind_old_h1 = [ind_no_term[kk],
                                  ind_no_term[kk]]

                    h2_new[ind_new_h2[0]][ind_new_h2[1]][
                        ind_new_h2[2]][ind_new_h2[3]] \
                        += 0.5 * temp_sign_h2 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                    h1_new[ind_old_h1[0]][ind_old_h1[1]] \
                        += 0.5 * temp_sign_h1 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                elif mapping_no_term == ['a', 'adag', 'a', 'adag']:

                    temp_sign_h2 = -1 * sign_no_term
                    temp_sign_h1_1 = -1 * sign_no_term
                    temp_sign_h1_2 = -1 * sign_no_term

                    ind_new_h2 = [ind_no_term[ii],
                                  ind_no_term[kk],
                                  ind_no_term[ii],
                                  ind_no_term[kk]]
                    ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                  ind_ini_term[2], ind_ini_term[3]]
                    ind_old_h2 = last_two_indices_swap(ind_old_h2)

                    coordinates_for_old_h1_term_1 = [ind_no_term[ii],
                                                     ind_no_term[ii]]
                    ind_old_h1_2 = [ind_no_term[kk],
                                    ind_no_term[kk]]

                    h2_new[ind_new_h2[0]][ind_new_h2[1]][
                        ind_new_h2[2]][ind_new_h2[3]] \
                        += 0.5 * temp_sign_h2 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                    h1_new[coordinates_for_old_h1_term_1[0]][coordinates_for_old_h1_term_1[1]] \
                        += 0.5 * temp_sign_h1_1 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                    h1_new[ind_old_h1_2[0]][ind_old_h1_2[1]] \
                        += 0.5 * temp_sign_h1_2 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                    id_term += 0.5 * sign_no_term * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                elif mapping_no_term == ['a', 'a', 'adag', 'adag']:

                    temp_sign_h2 = 1 * sign_no_term

                    ind_new_h2 = [ind_no_term[kk],
                                  ind_no_term[kk],
                                  ind_no_term[ii],
                                  ind_no_term[ii]]
                    ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                  ind_ini_term[2], ind_ini_term[3]]
                    ind_old_h2 = last_two_indices_swap(ind_old_h2)

                    h2_new[ind_new_h2[0]][ind_new_h2[1]][
                        ind_new_h2[2]][ind_new_h2[3]] \
                        += 0.5 * temp_sign_h2 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]
                else:
                    print('ERROR')

            elif ind_no_term[0] == ind_no_term[2] and \
                    ind_no_term[1] == ind_no_term[3]:

                if mapping_no_term == ['adag', 'adag', 'a', 'a']:

                    temp_sign_h2 = 1 * sign_no_term

                    ind_new_h2 = [ind_no_term[ii],
                                  ind_no_term[jj],
                                  ind_no_term[ii],
                                  ind_no_term[jj]]
                    ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                  ind_ini_term[2], ind_ini_term[3]]
                    ind_old_h2 = last_two_indices_swap(ind_old_h2)

                    h2_new[ind_new_h2[0]][ind_new_h2[1]][
                        ind_new_h2[2]][ind_new_h2[3]] \
                        += 0.5 * temp_sign_h2 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                elif mapping_no_term == ['adag', 'a', 'adag', 'a']:

                    temp_sign_h2 = -1 * sign_no_term

                    ind_new_h2 = [ind_no_term[ii],
                                  ind_no_term[ii],
                                  ind_no_term[jj],
                                  ind_no_term[jj]]
                    ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                  ind_ini_term[2], ind_ini_term[3]]
                    ind_old_h2 = last_two_indices_swap(ind_old_h2)

                    h2_new[ind_new_h2[0]][ind_new_h2[1]][
                        ind_new_h2[2]][ind_new_h2[3]] \
                        += 0.5 * temp_sign_h2 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                elif mapping_no_term == ['adag', 'a', 'a', 'adag']:

                    temp_sign_h2 = 1 * sign_no_term
                    temp_sign_h1 = -1 * sign_no_term

                    ind_new_h2 = [ind_no_term[ii],
                                  ind_no_term[jj],
                                  ind_no_term[jj],
                                  ind_no_term[ii]]
                    ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                  ind_ini_term[2], ind_ini_term[3]]
                    ind_old_h2 = last_two_indices_swap(ind_old_h2)

                    ind_old_h1 = [ind_no_term[ii],
                                  ind_no_term[ii]]

                    h2_new[ind_new_h2[0]][ind_new_h2[1]][
                        ind_new_h2[2]][ind_new_h2[3]] \
                        += 0.5 * temp_sign_h2 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                    h1_new[ind_old_h1[0]][ind_old_h1[1]] \
                        += 0.5 * temp_sign_h1 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                elif mapping_no_term == ['a', 'adag', 'adag', 'a']:

                    temp_sign_h2 = 1 * sign_no_term
                    temp_sign_h1 = -1 * sign_no_term

                    ind_new_h2 = [ind_no_term[jj],
                                  ind_no_term[ii],
                                  ind_no_term[ii],
                                  ind_no_term[jj]]
                    ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                  ind_ini_term[2], ind_ini_term[3]]
                    ind_old_h2 = last_two_indices_swap(ind_old_h2)

                    ind_old_h1 = [ind_no_term[jj],
                                  ind_no_term[jj]]

                    h2_new[ind_new_h2[0]][ind_new_h2[1]][
                        ind_new_h2[2]][ind_new_h2[3]] \
                        += 0.5 * temp_sign_h2 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                    h1_new[ind_old_h1[0]][ind_old_h1[1]] \
                        += 0.5 * temp_sign_h1 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                elif mapping_no_term == ['a', 'adag', 'a', 'adag']:

                    temp_sign_h2 = -1 * sign_no_term

                    ind_new_h2 = [ind_no_term[jj],
                                  ind_no_term[jj],
                                  ind_no_term[ii],
                                  ind_no_term[ii]]
                    ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                  ind_ini_term[2], ind_ini_term[3]]
                    ind_old_h2 = last_two_indices_swap(ind_old_h2)

                    h2_new[ind_new_h2[0]][ind_new_h2[1]][
                        ind_new_h2[2]][ind_new_h2[3]] \
                        += 0.5 * temp_sign_h2 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                elif mapping_no_term == ['a', 'a', 'adag', 'adag']:

                    temp_sign_h2 = 1 * sign_no_term
                    temp_sign_h1_1 = 1 * sign_no_term
                    temp_sign_h1_2 = 1 * sign_no_term

                    ind_new_h2 = [ind_no_term[ii],
                                  ind_no_term[jj],
                                  ind_no_term[ii],
                                  ind_no_term[jj]]
                    ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                  ind_ini_term[2], ind_ini_term[3]]
                    ind_old_h2 = last_two_indices_swap(ind_old_h2)

                    coordinates_for_old_h1_term_1 = [ind_no_term[ii],
                                                     ind_no_term[ii]]
                    ind_old_h1_2 = [ind_no_term[jj],
                                    ind_no_term[jj]]

                    h2_new[ind_new_h2[0]][ind_new_h2[1]][
                        ind_new_h2[2]][ind_new_h2[3]] \
                        += 0.5 * temp_sign_h2 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                    h1_new[coordinates_for_old_h1_term_1[0]][coordinates_for_old_h1_term_1[1]] \
                        += 0.5 * temp_sign_h1_1 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                    h1_new[ind_old_h1_2[0]][ind_old_h1_2[1]] \
                        += 0.5 * temp_sign_h1_2 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                    id_term += - 0.5 * sign_no_term * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]

                else:
                    print('ERROR')

            elif ind_no_term[0] == ind_no_term[3] and \
                    ind_no_term[1] == ind_no_term[2]:

                if mapping_no_term == ['adag', 'adag', 'a', 'a']:

                    temp_sign_h2 = 1 * sign_no_term

                    ind_new_h2 = [ind_no_term[ii],
                                  ind_no_term[jj],
                                  ind_no_term[jj],
                                  ind_no_term[ii]]
                    ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                  ind_ini_term[2], ind_ini_term[3]]
                    ind_old_h2 = last_two_indices_swap(ind_old_h2)

                    h2_new[ind_new_h2[0]][ind_new_h2[1]][
                        ind_new_h2[2]][ind_new_h2[3]] \
                        += 0.5 * temp_sign_h2 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                elif mapping_no_term == ['adag', 'a', 'adag', 'a']:

                    temp_sign_h2 = -1 * sign_no_term
                    temp_sign_h1 = 1 * sign_no_term

                    ind_new_h2 = [ind_no_term[ii],
                                  ind_no_term[jj],
                                  ind_no_term[jj],
                                  ind_no_term[ii]]
                    ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                  ind_ini_term[2], ind_ini_term[3]]
                    ind_old_h2 = last_two_indices_swap(ind_old_h2)

                    ind_old_h1 = [ind_no_term[ii],
                                  ind_no_term[ii]]

                    h2_new[ind_new_h2[0]][ind_new_h2[1]][
                        ind_new_h2[2]][ind_new_h2[3]] \
                        += 0.5 * temp_sign_h2 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                    h1_new[ind_old_h1[0]][ind_old_h1[1]] \
                        += 0.5 * temp_sign_h1 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                elif mapping_no_term == ['adag', 'a', 'a', 'adag']:

                    temp_sign_h2 = 1 * sign_no_term

                    ind_new_h2 = [ind_no_term[ii],
                                  ind_no_term[ii],
                                  ind_no_term[jj],
                                  ind_no_term[jj]]
                    ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                  ind_ini_term[2], ind_ini_term[3]]
                    ind_old_h2 = last_two_indices_swap(ind_old_h2)

                    h2_new[ind_new_h2[0]][ind_new_h2[1]][
                        ind_new_h2[2]][ind_new_h2[3]] \
                        += 0.5 * temp_sign_h2 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                elif mapping_no_term == ['a', 'adag', 'adag', 'a']:

                    temp_sign_h2 = 1 * sign_no_term

                    ind_new_h2 = [ind_no_term[jj],
                                  ind_no_term[jj],
                                  ind_no_term[ii],
                                  ind_no_term[ii]]
                    ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                  ind_ini_term[2], ind_ini_term[3]]
                    ind_old_h2 = last_two_indices_swap(ind_old_h2)

                    h2_new[ind_new_h2[0]][ind_new_h2[1]][
                        ind_new_h2[2]][ind_new_h2[3]] \
                        += 0.5 * temp_sign_h2 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                elif mapping_no_term == ['a', 'adag', 'a', 'adag']:

                    temp_sign_h2 = -1 * sign_no_term
                    temp_sign_h1 = 1 * sign_no_term

                    ind_new_h2 = [ind_no_term[jj],
                                  ind_no_term[ii],
                                  ind_no_term[ii],
                                  ind_no_term[jj]]
                    ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                  ind_ini_term[2], ind_ini_term[3]]
                    ind_old_h2 = last_two_indices_swap(ind_old_h2)

                    ind_old_h1 = [ind_no_term[jj],
                                  ind_no_term[jj]]

                    h2_new[ind_new_h2[0]][ind_new_h2[1]][
                        ind_new_h2[2]][ind_new_h2[3]] \
                        += 0.5 * temp_sign_h2 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                    h1_new[ind_old_h1[0]][ind_old_h1[1]] \
                        += 0.5 * temp_sign_h1 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                elif mapping_no_term == ['a', 'a', 'adag', 'adag']:

                    temp_sign_h2 = 1 * sign_no_term
                    temp_sign_h1_1 = -1 * sign_no_term
                    temp_sign_h1_2 = -1 * sign_no_term

                    ind_new_h2 = [ind_no_term[jj],
                                  ind_no_term[ii],
                                  ind_no_term[ii],
                                  ind_no_term[jj]]
                    ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                  ind_ini_term[2], ind_ini_term[3]]
                    ind_old_h2 = last_two_indices_swap(ind_old_h2)

                    coordinates_for_old_h1_term_1 = [ind_no_term[ii],
                                                     ind_no_term[ii]]
                    ind_old_h1_2 = [ind_no_term[jj],
                                    ind_no_term[jj]]

                    h2_new[ind_new_h2[0]][ind_new_h2[1]][
                        ind_new_h2[2]][ind_new_h2[3]] \
                        += 0.5 * temp_sign_h2 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                    h1_new[coordinates_for_old_h1_term_1[0]][coordinates_for_old_h1_term_1[1]] \
                        += 0.5 * temp_sign_h1_1 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                    h1_new[ind_old_h1_2[0]][ind_old_h1_2[1]] \
                        += 0.5 * temp_sign_h1_2 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                    id_term += 0.5 * sign_no_term * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][ind_old_h2[2]][ind_old_h2[3]]
                else:
                    print('ERROR')

            elif ind_no_term[0] == ind_no_term[1] and \
                    ind_no_term[0] == ind_no_term[2]:

                if mapping_no_term == ['adag', 'adag', 'a', 'a']:
                    temp_sign_h2 = 1 * sign_no_term

                    ind_new_h2 = [ind_no_term[ii],
                                  ind_no_term[ii],
                                  ind_no_term[ii],
                                  ind_no_term[ll]]
                    ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                  ind_ini_term[2], ind_ini_term[3]]
                    ind_old_h2 = last_two_indices_swap(ind_old_h2)

                    h2_new[ind_new_h2[0]][ind_new_h2[1]][
                        ind_new_h2[2]][ind_new_h2[3]] \
                        += 0.5 * temp_sign_h2 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                elif mapping_no_term == ['adag', 'a', 'adag', 'a']:

                    temp_sign_h2 = -1 * sign_no_term
                    temp_sign_h1_1 = 1 * sign_no_term

                    ind_new_h2 = [ind_no_term[ii],
                                  ind_no_term[ii],
                                  ind_no_term[ii],
                                  ind_no_term[ll]]
                    ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                  ind_ini_term[2], ind_ini_term[3]]
                    ind_old_h2 = last_two_indices_swap(ind_old_h2)

                    coordinates_for_old_h1_term_1 = [ind_no_term[ii],
                                                     ind_no_term[ll]]

                    h2_new[ind_new_h2[0]][ind_new_h2[1]][
                        ind_new_h2[2]][ind_new_h2[3]] \
                        += 0.5 * temp_sign_h2 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                    h1_new[coordinates_for_old_h1_term_1[0]][coordinates_for_old_h1_term_1[1]] \
                        += 0.5 * temp_sign_h1_1 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                elif mapping_no_term == ['adag', 'a', 'a', 'adag']:

                    temp_sign_h2 = 1 * sign_no_term

                    ind_new_h2 = [ind_no_term[ii],
                                  ind_no_term[ll],
                                  ind_no_term[ii],
                                  ind_no_term[ii]]
                    ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                  ind_ini_term[2], ind_ini_term[3]]
                    ind_old_h2 = last_two_indices_swap(ind_old_h2)

                    h2_new[ind_new_h2[0]][ind_new_h2[1]][
                        ind_new_h2[2]][ind_new_h2[3]] \
                        += 0.5 * temp_sign_h2 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                elif mapping_no_term == ['a', 'adag', 'adag', 'a']:

                    temp_sign_h2 = 1 * sign_no_term

                    ind_new_h2 = [ind_no_term[ii],
                                  ind_no_term[ii],
                                  ind_no_term[ii],
                                  ind_no_term[ll]]
                    ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                  ind_ini_term[2], ind_ini_term[3]]
                    ind_old_h2 = last_two_indices_swap(ind_old_h2)

                    h2_new[ind_new_h2[0]][ind_new_h2[1]][
                        ind_new_h2[2]][ind_new_h2[3]] \
                        += 0.5 * temp_sign_h2 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                elif mapping_no_term == ['a', 'adag', 'a', 'adag']:

                    temp_sign_h2 = -1 * sign_no_term
                    temp_sign_h1_1 = -1 * sign_no_term

                    ind_new_h2 = [ind_no_term[ii],
                                  ind_no_term[ll],
                                  ind_no_term[ii],
                                  ind_no_term[ii]]
                    ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                  ind_ini_term[2], ind_ini_term[3]]
                    ind_old_h2 = last_two_indices_swap(ind_old_h2)

                    coordinates_for_old_h1_term_1 = [ind_no_term[ll],
                                                     ind_no_term[ii]]

                    h2_new[ind_new_h2[0]][ind_new_h2[1]][
                        ind_new_h2[2]][ind_new_h2[3]] \
                        += 0.5 * temp_sign_h2 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                    h1_new[coordinates_for_old_h1_term_1[0]][coordinates_for_old_h1_term_1[1]] \
                        += 0.5 * temp_sign_h1_1 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                elif mapping_no_term == ['a', 'a', 'adag', 'adag']:

                    temp_sign_h2 = 1 * sign_no_term

                    ind_new_h2 = [ind_no_term[ii],
                                  ind_no_term[ll],
                                  ind_no_term[ii],
                                  ind_no_term[ii]]
                    ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                  ind_ini_term[2], ind_ini_term[3]]
                    ind_old_h2 = last_two_indices_swap(ind_old_h2)

                    h2_new[ind_new_h2[0]][ind_new_h2[1]][
                        ind_new_h2[2]][ind_new_h2[3]] \
                        += 0.5 * temp_sign_h2 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                else:
                    print('ERROR')

            elif ind_no_term[0] == ind_no_term[1] and \
                    ind_no_term[0] == ind_no_term[3]:

                if mapping_no_term == ['adag', 'adag', 'a', 'a']:

                    temp_sign_h2 = 1 * sign_no_term

                    ind_new_h2 = [ind_no_term[ii],
                                  ind_no_term[ii],
                                  ind_no_term[ii],
                                  ind_no_term[kk]]
                    ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                  ind_ini_term[2], ind_ini_term[3]]
                    ind_old_h2 = last_two_indices_swap(ind_old_h2)

                    h2_new[ind_new_h2[0]][ind_new_h2[1]][
                        ind_new_h2[2]][ind_new_h2[3]] \
                        += 0.5 * temp_sign_h2 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                elif mapping_no_term == ['adag', 'a', 'adag', 'a']:

                    temp_sign_h2 = -1 * sign_no_term
                    temp_sign_h1 = 1 * sign_no_term

                    ind_new_h2 = [ind_no_term[ii],
                                  ind_no_term[ii],
                                  ind_no_term[ii],
                                  ind_no_term[kk]]
                    ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                  ind_ini_term[2], ind_ini_term[3]]
                    ind_old_h2 = last_two_indices_swap(ind_old_h2)

                    ind_old_h1 = [ind_no_term[ii],
                                  ind_no_term[kk]]

                    h2_new[ind_new_h2[0]][ind_new_h2[1]][
                        ind_new_h2[2]][ind_new_h2[3]] \
                        += 0.5 * temp_sign_h2 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                    h1_new[ind_old_h1[0]][ind_old_h1[1]] \
                        += 0.5 * temp_sign_h1 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                elif mapping_no_term == ['adag', 'a', 'a', 'adag']:

                    temp_sign_h2 = 1 * sign_no_term

                    ind_new_h2 = [ind_no_term[ii],
                                  ind_no_term[kk],
                                  ind_no_term[ii],
                                  ind_no_term[ii]]
                    ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                  ind_ini_term[2], ind_ini_term[3]]
                    ind_old_h2 = last_two_indices_swap(ind_old_h2)

                    h2_new[ind_new_h2[0]][ind_new_h2[1]][
                        ind_new_h2[2]][ind_new_h2[3]] \
                        += 0.5 * temp_sign_h2 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                elif mapping_no_term == ['a', 'adag', 'adag', 'a']:

                    temp_sign_h2 = 1 * sign_no_term

                    ind_new_h2 = [ind_no_term[ii],
                                  ind_no_term[ii],
                                  ind_no_term[ii],
                                  ind_no_term[kk]]
                    ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                  ind_ini_term[2], ind_ini_term[3]]
                    ind_old_h2 = last_two_indices_swap(ind_old_h2)

                    h2_new[ind_new_h2[0]][ind_new_h2[1]][
                        ind_new_h2[2]][ind_new_h2[3]] \
                        += 0.5 * temp_sign_h2 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                elif mapping_no_term == ['a', 'adag', 'a', 'adag']:

                    temp_sign_h2 = -1 * sign_no_term
                    temp_sign_h1 = -1 * sign_no_term

                    ind_new_h2 = [ind_no_term[ii],
                                  ind_no_term[kk],
                                  ind_no_term[ii],
                                  ind_no_term[ii]]
                    ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                  ind_ini_term[2], ind_ini_term[3]]
                    ind_old_h2 = last_two_indices_swap(ind_old_h2)

                    ind_old_h1 = [ind_no_term[kk],
                                  ind_no_term[ii]]

                    h2_new[ind_new_h2[0]][ind_new_h2[1]][
                        ind_new_h2[2]][ind_new_h2[3]] \
                        += 0.5 * temp_sign_h2 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                    h1_new[ind_old_h1[0]][ind_old_h1[1]] \
                        += 0.5 * temp_sign_h1 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                elif mapping_no_term == ['a', 'a', 'adag', 'adag']:

                    temp_sign_h2 = 1 * sign_no_term

                    ind_new_h2 = [ind_no_term[ii],
                                  ind_no_term[kk],
                                  ind_no_term[ii],
                                  ind_no_term[ii]]
                    ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                  ind_ini_term[2], ind_ini_term[3]]
                    ind_old_h2 = last_two_indices_swap(ind_old_h2)

                    h2_new[ind_new_h2[0]][ind_new_h2[1]][
                        ind_new_h2[2]][ind_new_h2[3]] \
                        += 0.5 * temp_sign_h2 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                else:
                    print('ERROR')

            elif ind_no_term[0] == ind_no_term[2] and \
                    ind_no_term[0] == ind_no_term[3]:

                if mapping_no_term == ['adag', 'adag', 'a', 'a']:

                    temp_sign_h2 = 1 * sign_no_term

                    ind_new_h2 = [ind_no_term[ii],
                                  ind_no_term[jj],
                                  ind_no_term[ii],
                                  ind_no_term[ii]]
                    ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                  ind_ini_term[2], ind_ini_term[3]]
                    ind_old_h2 = last_two_indices_swap(ind_old_h2)

                    h2_new[ind_new_h2[0]][ind_new_h2[1]][
                        ind_new_h2[2]][ind_new_h2[3]] \
                        += 0.5 * temp_sign_h2 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                elif mapping_no_term == ['adag', 'a', 'adag', 'a']:

                    temp_sign_h2 = -1 * sign_no_term

                    ind_new_h2 = [ind_no_term[ii],
                                  ind_no_term[ii],
                                  ind_no_term[jj],
                                  ind_no_term[ii]]
                    ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                  ind_ini_term[2], ind_ini_term[3]]
                    ind_old_h2 = last_two_indices_swap(ind_old_h2)

                    h2_new[ind_new_h2[0]][ind_new_h2[1]][
                        ind_new_h2[2]][ind_new_h2[3]] \
                        += 0.5 * temp_sign_h2 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                elif mapping_no_term == ['adag', 'a', 'a', 'adag']:

                    temp_sign_h2 = 1 * sign_no_term
                    temp_sign_h1 = 1 * sign_no_term

                    ind_new_h2 = [ind_no_term[ii],
                                  ind_no_term[ii],
                                  ind_no_term[jj],
                                  ind_no_term[ii]]
                    ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                  ind_ini_term[2], ind_ini_term[3]]
                    ind_old_h2 = last_two_indices_swap(ind_old_h2)

                    ind_old_h1 = [ind_no_term[ii],
                                  ind_no_term[jj]]

                    h2_new[ind_new_h2[0]][ind_new_h2[1]][
                        ind_new_h2[2]][ind_new_h2[3]] \
                        += 0.5 * temp_sign_h2 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                    h1_new[ind_old_h1[0]][ind_old_h1[1]] \
                        += 0.5 * temp_sign_h1 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                elif mapping_no_term == ['a', 'adag', 'adag', 'a']:

                    temp_sign_h2 = -1 * sign_no_term
                    temp_sign_h1 = -1 * sign_no_term

                    ind_new_h2 = [ind_no_term[jj],
                                  ind_no_term[ii],
                                  ind_no_term[ii],
                                  ind_no_term[ii]]
                    ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                  ind_ini_term[2], ind_ini_term[3]]
                    ind_old_h2 = last_two_indices_swap(ind_old_h2)

                    ind_old_h1 = [ind_no_term[jj],
                                  ind_no_term[ii]]

                    h2_new[ind_new_h2[0]][ind_new_h2[1]][
                        ind_new_h2[2]][ind_new_h2[3]] \
                        += 0.5 * temp_sign_h2 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                    h1_new[ind_old_h1[0]][ind_old_h1[1]] \
                        += 0.5 * temp_sign_h1 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                elif mapping_no_term == ['a', 'adag', 'a', 'adag']:

                    temp_sign_h2 = -1 * sign_no_term

                    ind_new_h2 = [ind_no_term[jj],
                                  ind_no_term[ii],
                                  ind_no_term[ii],
                                  ind_no_term[ii]]
                    ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                  ind_ini_term[2], ind_ini_term[3]]
                    ind_old_h2 = last_two_indices_swap(ind_old_h2)

                    h2_new[ind_new_h2[0]][ind_new_h2[1]][
                        ind_new_h2[2]][ind_new_h2[3]] \
                        += 0.5 * temp_sign_h2 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                elif mapping_no_term == ['a', 'a', 'adag', 'adag']:

                    temp_sign_h2 = 1 * sign_no_term

                    ind_new_h2 = [ind_no_term[ii],
                                  ind_no_term[ii],
                                  ind_no_term[ii],
                                  ind_no_term[jj]]
                    ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                  ind_ini_term[2], ind_ini_term[3]]
                    ind_old_h2 = last_two_indices_swap(ind_old_h2)

                    h2_new[ind_new_h2[0]][ind_new_h2[1]][
                        ind_new_h2[2]][ind_new_h2[3]] \
                        += 0.5 * temp_sign_h2 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]
                else:
                    print('ERROR')

            elif ind_no_term[1] == ind_no_term[2] and \
                    ind_no_term[1] == ind_no_term[3]:

                if mapping_no_term == ['adag', 'adag', 'a', 'a']:

                    temp_sign_h2 = 1 * sign_no_term

                    ind_new_h2 = [ind_no_term[ii],
                                  ind_no_term[jj],
                                  ind_no_term[jj],
                                  ind_no_term[jj]]
                    ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                  ind_ini_term[2], ind_ini_term[3]]
                    ind_old_h2 = last_two_indices_swap(ind_old_h2)

                    h2_new[ind_new_h2[0]][ind_new_h2[1]][
                        ind_new_h2[2]][ind_new_h2[3]] \
                        += 0.5 * temp_sign_h2 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                elif mapping_no_term == ['adag', 'a', 'adag', 'a']:

                    temp_sign_h2 = -1 * sign_no_term
                    temp_sign_h1 = 1 * sign_no_term

                    ind_new_h2 = [ind_no_term[ii],
                                  ind_no_term[jj],
                                  ind_no_term[jj],
                                  ind_no_term[jj]]
                    ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                  ind_ini_term[2], ind_ini_term[3]]
                    ind_old_h2 = last_two_indices_swap(ind_old_h2)

                    ind_old_h1 = [ind_no_term[ii],
                                  ind_no_term[jj]]

                    h2_new[ind_new_h2[0]][ind_new_h2[1]][
                        ind_new_h2[2]][ind_new_h2[3]] \
                        += 0.5 * temp_sign_h2 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                    h1_new[ind_old_h1[0]][ind_old_h1[1]] \
                        += 0.5 * temp_sign_h1 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                elif mapping_no_term == ['adag', 'a', 'a', 'adag']:

                    temp_sign_h2 = 1 * sign_no_term

                    ind_new_h2 = [ind_no_term[ii],
                                  ind_no_term[jj],
                                  ind_no_term[jj],
                                  ind_no_term[jj]]
                    ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                  ind_ini_term[2], ind_ini_term[3]]
                    ind_old_h2 = last_two_indices_swap(ind_old_h2)

                    h2_new[ind_new_h2[0]][ind_new_h2[1]][
                        ind_new_h2[2]][ind_new_h2[3]] \
                        += 0.5 * temp_sign_h2 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                elif mapping_no_term == ['a', 'adag', 'adag', 'a']:

                    temp_sign_h2 = 1 * sign_no_term

                    ind_new_h2 = [ind_no_term[jj],
                                  ind_no_term[jj],
                                  ind_no_term[ii],
                                  ind_no_term[jj]]
                    ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                  ind_ini_term[2], ind_ini_term[3]]
                    ind_old_h2 = last_two_indices_swap(ind_old_h2)

                    h2_new[ind_new_h2[0]][ind_new_h2[1]][
                        ind_new_h2[2]][ind_new_h2[3]] \
                        += 0.5 * temp_sign_h2 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                elif mapping_no_term == ['a', 'adag', 'a', 'adag']:

                    temp_sign_h2 = -1 * sign_no_term
                    temp_sign_h1 = -1 * sign_no_term

                    ind_new_h2 = [ind_no_term[jj],
                                  ind_no_term[jj],
                                  ind_no_term[ii],
                                  ind_no_term[jj]]
                    ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                  ind_ini_term[2], ind_ini_term[3]]
                    ind_old_h2 = last_two_indices_swap(ind_old_h2)

                    ind_old_h1 = [ind_no_term[jj],
                                  ind_no_term[ii]]

                    h2_new[ind_new_h2[0]][ind_new_h2[1]][
                        ind_new_h2[2]][ind_new_h2[3]] \
                        += 0.5 * temp_sign_h2 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                    h1_new[ind_old_h1[0]][ind_old_h1[1]] \
                        += 0.5 * temp_sign_h1 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                elif mapping_no_term == ['a', 'a', 'adag', 'adag']:

                    temp_sign_h2 = 1 * sign_no_term

                    ind_new_h2 = [ind_no_term[jj],
                                  ind_no_term[jj],
                                  ind_no_term[ii],
                                  ind_no_term[jj]]
                    ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                                  ind_ini_term[2], ind_ini_term[3]]
                    ind_old_h2 = last_two_indices_swap(ind_old_h2)

                    h2_new[ind_new_h2[0]][ind_new_h2[1]][
                        ind_new_h2[2]][ind_new_h2[3]] \
                        += 0.5 * temp_sign_h2 * \
                        h2_old[ind_old_h2[0]][ind_old_h2[1]][
                        ind_old_h2[2]][ind_old_h2[3]]

                else:
                    print('ERROR')

            else:
                print('ERROR')

        if len(set(ind_no_term)) == 1:

            if mapping_no_term == ['adag', 'adag', 'a', 'a']:

                temp_sign_h2 = 1 * sign_no_term

                ind_new_h2 = [ind_no_term[ii], ind_no_term[ii],
                              ind_no_term[ii], ind_no_term[ii]]
                ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                              ind_ini_term[2], ind_ini_term[3]]
                ind_old_h2 = last_two_indices_swap(ind_old_h2)

                h2_new[ind_new_h2[0]][ind_new_h2[1]][
                    ind_new_h2[2]][ind_new_h2[3]] \
                    += 0.5 * temp_sign_h2 * \
                    h2_old[ind_old_h2[0]][ind_old_h2[1]][
                    ind_old_h2[2]][ind_old_h2[3]]

            elif mapping_no_term == ['a', 'a', 'adag', 'adag']:

                temp_sign_h2 = 1 * sign_no_term

                ind_new_h2 = [ind_no_term[ii], ind_no_term[ii],
                              ind_no_term[ii], ind_no_term[ii]]
                ind_old_h2 = [ind_ini_term[0], ind_ini_term[1],
                              ind_ini_term[2], ind_ini_term[3]]
                ind_old_h2 = last_two_indices_swap(ind_old_h2)

                h2_new[ind_new_h2[0]][ind_new_h2[1]][
                    ind_new_h2[2]][ind_new_h2[3]] \
                    += 0.5 * temp_sign_h2 * \
                    h2_old[ind_old_h2[0]][ind_old_h2[1]][
                    ind_old_h2[2]][ind_old_h2[3]]

            else:
                print('ERROR')

    return h1_new, h2_new, id_term


def particle_hole_transformation(n_qubits, n_occupied, h1_old_matrix, h2_old_matrix):
    """
    This function produces the necessary h1, h2, identity for work with Fermionic Operators script.

    Args:
        n_qubits (int): number of qubits
        n_occupied (int): number of electrons
        h1_old_matrix (numpy.ndarray): rFs terms from Gaussian
        h2_old_matrix (numpy.ndarray): rsgtu terms from Gaussian

    Returns:
        numpy.ndarray, numpy.ndarray, float: h1_prime, h2_prime, identities
    """
    h1_new_sum = np.zeros([n_qubits, n_qubits])
    h2_new_sum = np.zeros([n_qubits, n_qubits, n_qubits, n_qubits])

    h2_old_matrix = -2*h2_old_matrix.copy()
    h2_old_matrix = np.einsum('IJKL->IKLJ', h2_old_matrix.copy())

    h1_old_matrix = h1_old_matrix.copy()

    for r in range(n_qubits):
        for s in range(n_qubits):
            for i in range(n_occupied):

                h1_old_matrix[r][s] += h2_old_matrix[r][i][s][i].copy() - h2_old_matrix[r][i][i][s].copy()

    identities_new_sum = 0

    for i in range(n_qubits):
        for j in range(n_qubits):

            indices_1 = [i, j]
            array_mapping_1 = ['adag', 'a']

            h1_new_matrix = np.zeros([n_qubits, n_qubits])
            h2_new_matrix = np.zeros([n_qubits, n_qubits, n_qubits, n_qubits])

            h1_new_matrix, h2_new_matrix, identities = normal_order_integrals(
                n_qubits, n_occupied, indices_1, array_mapping_1, h1_old_matrix, h2_old_matrix, h1_new_matrix, h2_new_matrix)

            h1_new_sum += h1_new_matrix
            h2_new_sum += h2_new_matrix
            identities_new_sum += identities

    for i in range(n_qubits):
        for j in range(n_qubits):
            for k in range(n_qubits):
                for l in range(n_qubits):

                    array_to_be_ordered = [i, j, k, l]

                    array_mapping_2 = ['adag', 'adag', 'a', 'a']

                    h1_new_matrix = np.zeros([n_qubits, n_qubits])
                    h2_new_matrix = np.zeros([n_qubits, n_qubits, n_qubits, n_qubits])

                    h1_new_matrix, h2_new_matrix, identities = normal_order_integrals(
                        n_qubits, n_occupied, array_to_be_ordered, array_mapping_2, h1_old_matrix, h2_old_matrix, h1_new_matrix, h2_new_matrix)

                    h1_new_sum += h1_new_matrix
                    h2_new_sum += h2_new_matrix
                    identities_new_sum += identities

    h2_new_sum = np.einsum('IKMJ->IJKM', h2_new_sum)

    return h1_new_sum, h2_new_sum, identities_new_sum
