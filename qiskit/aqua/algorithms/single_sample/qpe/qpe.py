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
"""
The Quantum Phase Estimation Algorithm.
"""

import logging

import numpy as np
from qiskit.quantum_info import Pauli

from qiskit.aqua.operators.legacy import op_converter
from qiskit.aqua.utils import get_subsystem_density_matrix
from qiskit.aqua.algorithms import QuantumAlgorithm
from qiskit.aqua.circuits import PhaseEstimationCircuit
from qiskit.aqua.operators import WeightedPauliOperator
from qiskit.aqua.operators import BaseOperator
from qiskit.aqua.components.initial_states import InitialState
from qiskit.aqua.components.iqfts import IQFT
from qiskit.aqua.utils.validation import validate_min, validate_in_set

logger = logging.getLogger(__name__)

# pylint: disable=invalid-name


class QPE(QuantumAlgorithm):
    """The Quantum Phase Estimation algorithm."""

    def __init__(
            self, operator: BaseOperator, state_in: InitialState,
            iqft: IQFT, num_time_slices: int = 1,
            num_ancillae: int = 1, expansion_mode: str = 'trotter',
            expansion_order: int = 1, shallow_circuit_concat: bool = False
    ) -> None:
        """

        Args:
            operator: the hamiltonian Operator object
            state_in: the InitialState component
                representing the initial quantum state
            iqft: the Inverse Quantum Fourier Transform component
            num_time_slices: the number of time slices, has a min. value of 1.
            num_ancillae: the number of ancillary qubits to use for the measurement,
                            has a min. value of 1.
            expansion_mode: the expansion mode (trotter|suzuki)
            expansion_order: the suzuki expansion order, has a min. value of 1.
            shallow_circuit_concat: indicate whether to use shallow
                (cheap) mode for circuit concatenation
        """
        validate_min('num_time_slices', num_time_slices, 1)
        validate_min('num_ancillae', num_ancillae, 1)
        validate_in_set('expansion_mode', expansion_mode, {'trotter', 'suzuki'})
        validate_min('expansion_order', expansion_order, 1)
        super().__init__()
        self._operator = op_converter.to_weighted_pauli_operator(operator.copy())
        self._num_ancillae = num_ancillae
        self._ret = {}

        self._ret['translation'] = sum([abs(p[0]) for p in self._operator.reorder_paulis()])
        self._ret['stretch'] = 0.5 / self._ret['translation']

        # translate the operator
        self._operator.simplify()
        translation_op = WeightedPauliOperator([
            [
                self._ret['translation'],
                Pauli(
                    np.zeros(self._operator.num_qubits),
                    np.zeros(self._operator.num_qubits)
                )
            ]
        ])
        translation_op.simplify()
        self._operator += translation_op
        self._pauli_list = self._operator.reorder_paulis()

        # stretch the operator
        for p in self._pauli_list:
            p[0] = p[0] * self._ret['stretch']

        self._phase_estimation_circuit = PhaseEstimationCircuit(
            operator=self._operator, state_in=state_in, iqft=iqft,
            num_time_slices=num_time_slices, num_ancillae=num_ancillae,
            expansion_mode=expansion_mode, expansion_order=expansion_order,
            shallow_circuit_concat=shallow_circuit_concat, pauli_list=self._pauli_list
        )
        self._binary_fractions = [1 / 2 ** p for p in range(1, num_ancillae + 1)]

    def construct_circuit(self, measurement=False):
        """
        Construct circuit.

        Args:
            measurement (bool): Boolean flag to indicate if measurement
                should be included in the circuit.

        Returns:
            QuantumCircuit: quantum circuit.
        """
        qc = self._phase_estimation_circuit.construct_circuit(measurement=measurement)
        return qc

    def _compute_energy(self):
        if self._quantum_instance.is_statevector:
            qc = self.construct_circuit(measurement=False)
            result = self._quantum_instance.execute(qc)
            complete_state_vec = result.get_statevector(qc)
            ancilla_density_mat = get_subsystem_density_matrix(
                complete_state_vec,
                range(self._num_ancillae, self._num_ancillae + self._operator.num_qubits)
            )
            ancilla_density_mat_diag = np.diag(ancilla_density_mat)
            max_amplitude = \
                max(ancilla_density_mat_diag.min(), ancilla_density_mat_diag.max(), key=abs)
            max_amplitude_idx = np.where(ancilla_density_mat_diag == max_amplitude)[0][0]
            top_measurement_label = np.binary_repr(max_amplitude_idx, self._num_ancillae)[::-1]
        else:
            qc = self.construct_circuit(measurement=True)
            result = self._quantum_instance.execute(qc)
            ancilla_counts = result.get_counts(qc)
            top_measurement_label = \
                sorted([(ancilla_counts[k], k) for k in ancilla_counts])[::-1][0][-1][::-1]

        top_measurement_decimal = sum(
            [t[0] * t[1] for t in zip(self._binary_fractions,
                                      [int(n) for n in top_measurement_label])]
        )

        self._ret['top_measurement_label'] = top_measurement_label
        self._ret['top_measurement_decimal'] = top_measurement_decimal
        self._ret['eigvals'] = \
            [top_measurement_decimal / self._ret['stretch'] - self._ret['translation']]
        self._ret['energy'] = self._ret['eigvals'][0]

    def _run(self):
        self._compute_energy()
        return self._ret
