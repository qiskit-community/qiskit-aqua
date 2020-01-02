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
"""
The Deutsch-Jozsa algorithm.
"""

import logging
import operator
import numpy as np

from qiskit import ClassicalRegister, QuantumCircuit

from qiskit.aqua.algorithms import QuantumAlgorithm
from qiskit.aqua.utils import get_subsystem_density_matrix
from qiskit.aqua.utils.validation import validate

logger = logging.getLogger(__name__)

# pylint: disable=invalid-name


class DeutschJozsa(QuantumAlgorithm):
    """The Deutsch-Jozsa algorithm."""

    _INPUT_SCHEMA = {
        '$schema': 'http://json-schema.org/draft-07/schema#',
        'id': 'dj_schema',
        'type': 'object',
        'properties': {
        },
        'additionalProperties': False
    }

    def __init__(self, oracle):
        validate(locals(), self._INPUT_SCHEMA)
        super().__init__()

        self._oracle = oracle
        self._circuit = None
        self._ret = {}

    def construct_circuit(self, measurement=False):
        """
        Construct the quantum circuit

        Args:
            measurement (bool): Boolean flag to indicate
                if measurement should be included in the circuit.

        Returns:
            QuantumCircuit: the QuantumCircuit object for the constructed circuit
        """

        if self._circuit is not None:
            return self._circuit

        # preoracle circuit
        qc_preoracle = QuantumCircuit(
            self._oracle.variable_register,
            self._oracle.output_register,
        )
        qc_preoracle.h(self._oracle.variable_register)
        qc_preoracle.x(self._oracle.output_register)
        qc_preoracle.h(self._oracle.output_register)
        qc_preoracle.barrier()

        # oracle circuit
        qc_oracle = self._oracle.circuit

        # postoracle circuit
        qc_postoracle = QuantumCircuit(
            self._oracle.variable_register,
            self._oracle.output_register,
        )
        qc_postoracle.h(self._oracle.variable_register)
        qc_postoracle.barrier()

        self._circuit = qc_preoracle + qc_oracle + qc_postoracle

        # measurement circuit
        if measurement:
            measurement_cr = ClassicalRegister(len(self._oracle.variable_register), name='m')
            self._circuit.add_register(measurement_cr)
            self._circuit.measure(self._oracle.variable_register, measurement_cr)

        return self._circuit

    def _run(self):
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

        self._ret['result'] = 'constant' if int(top_measurement) == 0 else 'balanced'

        return self._ret
