# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Convert knapsack parameters instances into Pauli list
The parameters are a list of values a list of weights and a maximum weight of the knapsack.

In the Knapsack Problem we are given a list of objects that each has a weight and a value.
We are also given a maximum weight we can carry. We need to pick a subset of the objects
so as to maximize the total value without going over the maximum weight.

If we have the weights w[i], the values v[i] and the maximum weight W_max.
We express the solution as a binary array x[i]
where we have a 1 for the items we take in the solution set.
We need to maximize sum(x[i]*v[i]) while respecting W_max >= sum(x[i]*w[i])

"""

import logging
import math
import numpy as np

from qiskit.quantum_info import Pauli
from qiskit.aqua.operators import WeightedPauliOperator


logger = logging.getLogger(__name__)


def get_operator(values, weights, max_weight):
    """
    Generate Hamiltonian for the knapsack problem.

    Notes:
        To build the cost function for the Hamiltonian we add a term S
        that will vary with our solution. In order to make it change wit the solution
        we enhance X with a number of additional bits X' = [x_0,..x_{n-1},y_{n}..y_{n+m-1}].
        The bytes y[i] will be the binary representation of S.
        In this way the optimizer will be able to optimize S as well as X.

        The cost function is
        $$C(X') = M(W_{max} - \\sum_{i=0}^{n-1} x_{i}w_{i} - S)^2 - \\sum_{i}^{n-1} x_{i}v_{i}$$
        where S = sum(2**j * y[j]), j goes from n to n+log(W_max).
        M is a number large enough to dominate the sum of values.

        Because S can only be positive, when W_max >= sum(x[i]*w[i])
        the optimizer can find an S (or better the y[j] that compose S)
        so that it will take the first term to 0.
        This way the function is dominated by the sum of values.
        If W_max < sum(x[i]*w[i]) then the first term can never be 0
        and, multiplied by a large M, will always dominate the function.

        The minimum value of the function will be that where the constraint is respected
        and the sum of values is maximized.

    Args:
        values (list of non-negative integers) : a list of values
        weights (list of non-negative integers) : a list of weights
        max_weight (non negative integer) : the maximum weight the knapsack can carry

    Returns:
        WeightedPauliOperator: operator for the Hamiltonian
        float: a constant shift for the obj function.

    Raises:
        ValueError: values and weights have different lengths
        ValueError: A value or a weight is negative
        ValueError: All values are zero
        ValueError: max_weight is negative

    """
    if len(values) != len(weights):
        raise ValueError("The values and weights must have the same length")

    if any(v < 0 for v in values) or any(w < 0 for w in weights):
        raise ValueError("The values and weights cannot be negative")

    if all(v == 0 for v in values):
        raise ValueError("The values cannot all be 0")

    if max_weight < 0:
        raise ValueError("max_weight cannot be negative")

    y_size = int(math.log(max_weight, 2)) + 1 if max_weight > 0 else 1
    n = len(values)
    num_values = n + y_size
    pauli_list = []
    shift = 0

    # pylint: disable=invalid-name
    M = 10 * np.sum(values)

    # term for sum(x_i*w_i)**2
    for i in range(n):
        for j in range(n):
            coefficient = -1 * 0.25 * weights[i] * weights[j] * M
            pauli_op = _get_pauli_op(num_values, [j])
            pauli_list.append([coefficient, pauli_op])
            shift -= coefficient

            pauli_op = _get_pauli_op(num_values, [i])
            pauli_list.append([coefficient, pauli_op])
            shift -= coefficient

            coefficient = -1 * coefficient
            pauli_op = _get_pauli_op(num_values, [i, j])
            pauli_list.append([coefficient, pauli_op])
            shift -= coefficient

    # term for sum(2**j*y_j)**2
    for i in range(y_size):
        for j in range(y_size):
            coefficient = -1 * 0.25 * (2 ** i) * (2 ** j) * M

            pauli_op = _get_pauli_op(num_values, [n + j])
            pauli_list.append([coefficient, pauli_op])
            shift -= coefficient

            pauli_op = _get_pauli_op(num_values, [n + i])
            pauli_list.append([coefficient, pauli_op])
            shift -= coefficient

            coefficient = -1 * coefficient
            pauli_op = _get_pauli_op(num_values, [n + i, n + j])
            pauli_list.append([coefficient, pauli_op])
            shift -= coefficient

    # term for -2*W_max*sum(x_i*w_i)
    for i in range(n):
        coefficient = max_weight * weights[i] * M

        pauli_op = _get_pauli_op(num_values, [i])
        pauli_list.append([coefficient, pauli_op])
        shift -= coefficient

    # term for -2*W_max*sum(2**j*y_j)
    for j in range(y_size):
        coefficient = max_weight * (2 ** j) * M

        pauli_op = _get_pauli_op(num_values, [n + j])
        pauli_list.append([coefficient, pauli_op])
        shift -= coefficient

    for i in range(n):
        for j in range(y_size):
            coefficient = -1 * 0.5 * weights[i] * (2 ** j) * M

            pauli_op = _get_pauli_op(num_values, [n + j])
            pauli_list.append([coefficient, pauli_op])
            shift -= coefficient

            pauli_op = _get_pauli_op(num_values, [i])
            pauli_list.append([coefficient, pauli_op])
            shift -= coefficient

            coefficient = -1 * coefficient
            pauli_op = _get_pauli_op(num_values, [i, n + j])
            pauli_list.append([coefficient, pauli_op])
            shift -= coefficient

    # term for sum(x_i*v_i)
    for i in range(n):
        coefficient = 0.5 * values[i]

        pauli_op = _get_pauli_op(num_values, [i])
        pauli_list.append([coefficient, pauli_op])
        shift -= coefficient

    return WeightedPauliOperator(paulis=pauli_list), shift


def get_solution(x, values):
    """
    Get the solution to the knapsack problem
    from the bitstring that represents
    to the ground state of the Hamiltonian

    Args:
        x (numpy.ndarray): the ground state of the Hamiltonian.
        values (numpy.ndarray): the list of values

    Returns:
        numpy.ndarray: a bit string that has a '1' at the indexes
         corresponding to values that have been taken in the knapsack.
         i.e. if the solution has a '1' at index i then
         the value values[i] has been taken in the knapsack
    """
    return x[:len(values)]


def knapsack_value_weight(solution, values, weights):
    """
    Get the total wight and value of the items taken in the knapsack.

    Args:
        solution (numpy.ndarray) : binary string that represents the solution to the problem.
        values (numpy.ndarray) : the list of values
        weights (numpy.ndarray) : the list of weights

    Returns:
        tuple: the total value and weight of the items in the knapsack
    """
    value = np.sum(solution * values)
    weight = np.sum(solution * weights)
    return value, weight


def _get_pauli_op(num_values, indexes):
    pauli_x = np.zeros(num_values, dtype=np.bool)
    pauli_z = np.zeros(num_values, dtype=np.bool)
    for i in indexes:
        pauli_z[i] = not pauli_z[i]

    return Pauli(pauli_z, pauli_x)
