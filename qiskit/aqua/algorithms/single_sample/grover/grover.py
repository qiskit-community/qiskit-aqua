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
The Grover's Search algorithm.
"""

from typing import Optional
import logging
import operator
import numpy as np

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.qasm import pi

from qiskit.aqua import AquaError
from qiskit.aqua.utils import get_subsystem_density_matrix
from qiskit.aqua.utils.validation import validate_min, validate_in_set
from qiskit.aqua.algorithms import QuantumAlgorithm
from qiskit.aqua.components.initial_states import Custom
from qiskit.aqua.components.oracles import Oracle
from qiskit.aqua.components.initial_states import InitialState
from qiskit.aqua.circuits.gates import mct  # pylint: disable=unused-import

logger = logging.getLogger(__name__)

# pylint: disable=invalid-name


class Grover(QuantumAlgorithm):
    """
    The Grover's Search algorithm.

    If the `num_iterations` param is specified, the amplitude amplification
    iteration will be built as specified.

    If the `incremental` mode is specified, which indicates that the optimal
    `num_iterations` isn't known in advance,
    a multi-round schedule will be followed with incremental trial `num_iterations` values.
    The implementation follows Section 4 of Boyer et al. <https://arxiv.org/abs/quant-ph/9605034>
    """

    def __init__(self, oracle: Oracle, init_state: Optional[InitialState] = None,
                 incremental: bool = False, num_iterations: int = 1,
                 mct_mode: str = 'basic') -> None:
        """
        Constructor.

        Args:
            oracle: the oracle component
            init_state: the initial quantum state preparation
            incremental: boolean flag for whether to use incremental search mode or not
            num_iterations: the number of iterations to use for amplitude amplification,
                            has a min. value of 1.
            mct_mode: mct mode
        Raises:
            AquaError: evaluate_classically() missing from the input oracle
        """
        validate_min('num_iterations', num_iterations, 1)
        validate_in_set('mct_mode', mct_mode,
                        {'basic', 'basic-dirty-ancilla',
                         'advanced', 'noancilla'})
        super().__init__()

        if not callable(getattr(oracle, "evaluate_classically", None)):
            raise AquaError(
                'Missing the evaluate_classically() method from the provided oracle instance.'
            )

        self._oracle = oracle
        self._mct_mode = mct_mode
        self._init_state = \
            init_state if init_state else Custom(len(oracle.variable_register), state='uniform')
        self._init_state_circuit = \
            self._init_state.construct_circuit(mode='circuit', register=oracle.variable_register)
        self._init_state_circuit_inverse = self._init_state_circuit.inverse()

        self._diffusion_circuit = self._construct_diffusion_circuit()
        self._max_num_iterations = np.ceil(2 ** (len(oracle.variable_register) / 2))
        self._incremental = incremental
        self._num_iterations = num_iterations if not incremental else 1
        if incremental:
            logger.debug('Incremental mode specified, ignoring "num_iterations".')
        else:
            if num_iterations > self._max_num_iterations:
                logger.warning('The specified value %s for "num_iterations" '
                               'might be too high.', num_iterations)
        self._ret = {}
        self._qc_aa_iteration = None
        self._qc_amplitude_amplification = None
        self._qc_measurement = None

    def _construct_diffusion_circuit(self):
        qc = QuantumCircuit(self._oracle.variable_register)
        num_variable_qubits = len(self._oracle.variable_register)
        num_ancillae_needed = 0
        if self._mct_mode == 'basic' or self._mct_mode == 'basic-dirty-ancilla':
            num_ancillae_needed = max(0, num_variable_qubits - 2)
        elif self._mct_mode == 'advanced' and num_variable_qubits >= 5:
            num_ancillae_needed = 1

        # check oracle's existing ancilla and add more if necessary
        num_oracle_ancillae = \
            len(self._oracle.ancillary_register) if self._oracle.ancillary_register else 0
        num_additional_ancillae = num_ancillae_needed - num_oracle_ancillae
        if num_additional_ancillae > 0:
            extra_ancillae = QuantumRegister(num_additional_ancillae, name='a_e')
            qc.add_register(extra_ancillae)
            ancilla = list(extra_ancillae)
            if num_oracle_ancillae > 0:
                ancilla += list(self._oracle.ancillary_register)
        else:
            ancilla = self._oracle.ancillary_register

        if self._oracle.ancillary_register:
            qc.add_register(self._oracle.ancillary_register)
        qc.barrier(self._oracle.variable_register)
        qc += self._init_state_circuit_inverse
        qc.u3(pi, 0, pi, self._oracle.variable_register)
        qc.u2(0, pi, self._oracle.variable_register[num_variable_qubits - 1])
        qc.mct(
            self._oracle.variable_register[0:num_variable_qubits - 1],
            self._oracle.variable_register[num_variable_qubits - 1],
            ancilla,
            mode=self._mct_mode
        )
        qc.u2(0, pi, self._oracle.variable_register[num_variable_qubits - 1])
        qc.u3(pi, 0, pi, self._oracle.variable_register)
        qc += self._init_state_circuit
        qc.barrier(self._oracle.variable_register)
        return qc

    @property
    def qc_amplitude_amplification_iteration(self):
        """ qc amplitude amplification iteration """
        if self._qc_aa_iteration is None:
            self._qc_aa_iteration = QuantumCircuit()
            self._qc_aa_iteration += self._oracle.circuit
            self._qc_aa_iteration += self._diffusion_circuit
        return self._qc_aa_iteration

    def _run_with_existing_iterations(self):
        if self._quantum_instance.is_statevector:
            qc = self.construct_circuit(measurement=False)
            result = self._quantum_instance.execute(qc)
            complete_state_vec = result.get_statevector(qc)
            variable_register_density_matrix = get_subsystem_density_matrix(
                complete_state_vec,
                range(len(self._oracle.variable_register), qc.width())
            )
            variable_register_density_matrix_diag = np.diag(variable_register_density_matrix)
            max_amplitude = max(
                variable_register_density_matrix_diag.min(),
                variable_register_density_matrix_diag.max(),
                key=abs
            )
            max_amplitude_idx = \
                np.where(variable_register_density_matrix_diag == max_amplitude)[0][0]
            top_measurement = np.binary_repr(max_amplitude_idx, len(self._oracle.variable_register))
        else:
            qc = self.construct_circuit(measurement=True)
            measurement = self._quantum_instance.execute(qc).get_counts(qc)
            self._ret['measurement'] = measurement
            top_measurement = max(measurement.items(), key=operator.itemgetter(1))[0]

        self._ret['top_measurement'] = top_measurement
        oracle_evaluation, assignment = self._oracle.evaluate_classically(top_measurement)
        return assignment, oracle_evaluation

    def construct_circuit(self, measurement=False):
        """
        Construct the quantum circuit

        Args:
            measurement (bool): Boolean flag to indicate if
                measurement should be included in the circuit.

        Returns:
            QuantumCircuit: the QuantumCircuit object for the constructed circuit
        """
        if self._qc_amplitude_amplification is None:
            self._qc_amplitude_amplification = \
                QuantumCircuit() + self.qc_amplitude_amplification_iteration
        qc = QuantumCircuit(self._oracle.variable_register, self._oracle.output_register)
        qc.u3(pi, 0, pi, self._oracle.output_register)  # x
        qc.u2(0, pi, self._oracle.output_register)  # h
        qc += self._init_state_circuit
        qc += self._qc_amplitude_amplification

        if measurement:
            measurement_cr = ClassicalRegister(len(self._oracle.variable_register), name='m')
            qc.add_register(measurement_cr)
            qc.measure(self._oracle.variable_register, measurement_cr)

        self._ret['circuit'] = qc
        return qc

    def _run(self):
        if self._incremental:
            current_max_num_iterations, lam = 1, 6 / 5

            def _try_current_max_num_iterations():
                target_num_iterations = self.random.randint(current_max_num_iterations) + 1
                self._qc_amplitude_amplification = QuantumCircuit()
                for _ in range(target_num_iterations):
                    self._qc_amplitude_amplification += self.qc_amplitude_amplification_iteration
                return self._run_with_existing_iterations()

            while current_max_num_iterations < self._max_num_iterations:
                assignment, oracle_evaluation = _try_current_max_num_iterations()
                if oracle_evaluation:
                    break
                current_max_num_iterations = \
                    min(lam * current_max_num_iterations, self._max_num_iterations)
        else:
            self._qc_amplitude_amplification = QuantumCircuit()
            for _ in range(self._num_iterations):
                self._qc_amplitude_amplification += self.qc_amplitude_amplification_iteration
            assignment, oracle_evaluation = self._run_with_existing_iterations()

        self._ret['result'] = assignment
        self._ret['oracle_evaluation'] = oracle_evaluation
        return self._ret
