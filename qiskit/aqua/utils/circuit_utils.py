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

""" Circuit utility functions """

from typing import Dict

import numpy as np

from qiskit import QuantumCircuit, transpile
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.quantum_info import Operator, Statevector
from qiskit.transpiler.passes import Unroller


def convert_to_basis_gates(circuit):
    """ unroll the circuit using the basis u1, u2, u3, cx gates """
    unroller = Unroller(basis=['u1', 'u2', 'u3', 'cx'])
    return dag_to_circuit(unroller.run(circuit_to_dag(circuit)))


def summarize_circuits(circuits):
    """Summarize circuits based on QuantumCircuit, and five metrics are summarized.
        - Number of qubits
        - Number of classical bits
        - Number of operations
        - Depth of circuits
        - Counts of different gate operations

    The average statistic of the first four is provided if multiple circuits are provided.

    Args:
        circuits (QuantumCircuit or [QuantumCircuit]): the to-be-summarized circuits

    Returns:
        str: a formatted string records the summary
    """
    if not isinstance(circuits, list):
        circuits = [circuits]
    ret = ""
    ret += "Submitting {} circuits.\n".format(len(circuits))
    ret += "============================================================================\n"
    stats = np.zeros(4)
    for i, circuit in enumerate(circuits):
        depth = circuit.depth()
        size = circuit.size()
        num_qubits = sum(reg.size for reg in circuit.qregs)
        num_clbits = sum(reg.size for reg in circuit.cregs)
        op_counts = circuit.count_ops()
        stats[0] += num_qubits
        stats[1] += num_clbits
        stats[2] += size
        stats[3] += depth
        ret = ''.join([
            ret,
            "{}-th circuit: {} qubits, {} classical bits and {} "
            "operations with depth {}\nop_counts: {}\n".format(
                i, num_qubits, num_clbits, size, depth, op_counts
            )
        ])
    if len(circuits) > 1:
        stats /= len(circuits)
        ret = ''.join([
            ret,
            "Average: {:.2f} qubits, {:.2f} classical bits and {:.2f} "
            "operations with depth {:.2f}\n".format(
                stats[0], stats[1], stats[2], stats[3]
            )
        ])
    ret += "============================================================================\n"
    return ret


class QuantumCircuitConverter:
    """Convert a quantum circuit into a unitary matrix, a state vector, and counts"""

    def __init__(self, qc: QuantumCircuit):
        """Initialize a converter with a circuit

        Note:
            It does not unroll all gates to 'u3' and 'cx' to deal with global phase.

        Args:
            qc: a quantum circuit
        """
        self._qc = transpile(qc,
                             basis_gates=['u1', 'u2', 'u3', 'cx', 'id', 'x', 'h', 'hamiltonian'])

    def _statevector(self):
        return Statevector.from_int(0, 2 ** self._qc.num_qubits).evolve(self._qc)

    def to_unitary_matrix(self) -> np.ndarray:
        """Return a unitary matrix corresponding to a quantum circuit.

        Returns: a unitary matrix.
        """
        return Operator(self._qc).data

    def to_state_vector(self) -> np.ndarray:
        """Return a state vector corresponding to a quantum circuit.

        Returns: a state vector.
        """
        return self._statevector().data

    def to_state_vector_dict(self) -> Dict[str, complex]:
        """Return a state vector dictionary corresponding to a quantum circuit.

        Returns: a state vector dictionary.
        """
        return self._statevector().to_dict()

    def to_counts(self, shots) -> Dict[str, int]:
        """Return counts corresponding to a quantum circuit.

        Returns: counts.
        """
        prob = self._statevector().probabilities_dict()
        keys = list(prob.keys())
        values = list(prob.values())
        indices, counts = np.unique(np.random.choice(a=len(prob), size=shots, p=values),
                                    return_counts=True)
        return {keys[i]: count for i, count in zip(indices, counts)}
