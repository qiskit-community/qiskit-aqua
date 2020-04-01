
# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Auxilliary methods to translate eigenvectors into optimization results."""

from qiskit.aqua.operators import MatrixOperator
from typing import Union, List, Tuple
from qiskit import QuantumCircuit, BasicAer, execute
from qiskit.aqua import QuantumInstance
from qiskit.aqua.operators import WeightedPauliOperator, MatrixOperator
import numpy


def eigenvector_to_solutions(eigenvector: Union[dict, numpy.ndarray],
                             operator: Union[WeightedPauliOperator, MatrixOperator],
                             min_probability: float = 1e-6) -> List[Tuple[str, float, float]]:
    """Convert the eigenvector to the bitstrings and corresponding eigenvalues.

    Examples:
    >>> op = MatrixOperator(numpy.array([[1, 1], [1, -1]]) / numpy.sqrt(2))
    >>> eigenvectors = {'0': 12, '1': 1}
    >>> print(eigenvector_to_solutions(eigenvectors, op))
    [('0', 0.7071067811865475, 0.9230769230769231), ('1', -0.7071067811865475, 0.07692307692307693)]

    >>> op = MatrixOperator(numpy.array([[1, 1], [1, -1]]) / numpy.sqrt(2))
    >>> eigenvectors = numpy.array([1, 1] / numpy.sqrt(2), dtype=complex)
    >>> print(eigenvector_to_solutions(eigenvectors, op))
    [('0', 0.7071067811865475, 0.4999999999999999), ('1', -0.7071067811865475, 0.4999999999999999)]

    Returns:
        For each computational basis state contained in the eigenvector, return the basis
        state as bitstring along with the operator evaluated at that bitstring and the
        probability of sampling this bitstring from the eigenvector.
    """
    solutions = []
    if isinstance(eigenvector, dict):
        all_counts = sum(eigenvector.values())
        # iterate over all samples
        for bitstr, count in eigenvector.items():
            sampling_probability = count / all_counts
            # add the bitstring, if the sampling probability exceeds the threshold
            if sampling_probability > 0:
                if sampling_probability >= min_probability:
                    value = eval_operator_at_bitstring(operator, bitstr)
                    solutions += [(bitstr, value, sampling_probability)]

    elif isinstance(eigenvector, numpy.ndarray):
        num_qubits = int(numpy.log2(eigenvector.size))
        probabilities = numpy.abs(eigenvector * eigenvector.conj())

        # iterate over all states and their sampling probabilities
        for i, sampling_probability in enumerate(probabilities):

            # add the i-th state if the sampling probability exceeds the threshold
            if sampling_probability > 0:
                if sampling_probability >= min_probability:
                    bitstr = '{:b}'.format(i).rjust(num_qubits, '0')[::-1]
                    value = eval_operator_at_bitstring(operator, bitstr)
                    solutions += [(bitstr, value, sampling_probability)]

    else:
        raise TypeError('Unsupported format of eigenvector. Provide a dict or numpy.ndarray.')

    return solutions


def eval_operator_at_bitstring(operator: Union[WeightedPauliOperator, MatrixOperator],
                               bitstr: str) -> float:
    """
    TODO
    """

    # TODO check that operator size and bitstr are compatible
    circuit = QuantumCircuit(len(bitstr))
    for i, bit in enumerate(bitstr):
        # TODO in which order to iterate over the bitstring???
        if bit == '1':
            circuit.x(i)

    # simulate the circuit
    result = execute(circuit, BasicAer.get_backend('statevector_simulator')).result()

    # evaluate the operator
    value = numpy.real(operator.evaluate_with_statevector(result.get_statevector())[0])

    return value
