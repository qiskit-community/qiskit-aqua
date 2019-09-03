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
Simon's algorithm.
"""

import operator  # pylint: disable=unused-import
import numpy as np
from sympy import Matrix, mod_inverse

from qiskit import ClassicalRegister, QuantumCircuit

from qiskit.aqua import AquaError, Pluggable, PluggableType, get_pluggable_class
from qiskit.aqua.algorithms import QuantumAlgorithm
from qiskit.aqua.utils import get_subsystem_density_matrix

# pylint: disable=invalid-name


class Simon(QuantumAlgorithm):
    """The Simon algorithm."""

    CONFIGURATION = {
        'name': 'Simon',
        'description': 'Simon',
        'input_schema': {
            '$schema': 'http://json-schema.org/draft-07/schema#',
            'id': 'simon_schema',
            'type': 'object',
            'properties': {
            },
            'additionalProperties': False
        },
        'problems': ['periodfinding'],
        'depends': [
            {
                'pluggable_type': 'oracle',
                'default': {
                    'name': 'TruthTableOracle',
                },
            },
        ],
    }

    def __init__(self, oracle):
        self.validate(locals())
        super().__init__()

        self._oracle = oracle
        self._circuit = None
        self._ret = {}

    @classmethod
    def init_params(cls, params, algo_input):
        """ init params """
        if algo_input is not None:
            raise AquaError("Input instance not supported.")

        oracle_params = params.get(Pluggable.SECTION_KEY_ORACLE)
        oracle = get_pluggable_class(
            PluggableType.ORACLE,
            oracle_params['name']).init_params(params)
        return cls(oracle)

    def construct_circuit(self, measurement=False):
        """
        Construct the quantum circuit

        Args:
            measurement (bool): Boolean flag to indicate if
                measurement should be included in the circuit.

        Returns:
            QuantumCircuit: the QuantumCircuit object for the constructed circuit
        """

        if self._circuit is not None:
            return self._circuit

        qc_preoracle = QuantumCircuit(
            self._oracle.variable_register,
            self._oracle.output_register,
        )
        qc_preoracle.h(self._oracle.variable_register)
        qc_preoracle.barrier()

        # oracle circuit
        qc_oracle = self._oracle.circuit
        qc_oracle.barrier()

        # postoracle circuit
        qc_postoracle = QuantumCircuit(
            self._oracle.variable_register,
            self._oracle.output_register,
        )
        qc_postoracle.h(self._oracle.variable_register)

        self._circuit = qc_preoracle + qc_oracle + qc_postoracle

        # measurement circuit
        if measurement:
            measurement_cr = ClassicalRegister(len(self._oracle.variable_register), name='m')
            self._circuit.add_register(measurement_cr)
            self._circuit.measure(self._oracle.variable_register, measurement_cr)

        return self._circuit

    def _interpret_measurement(self, measurements):
        # reverse measurement bitstrings and remove all zero entry
        linear = [(k[::-1], v) for k, v in measurements.items()
                  if k != "0" * len(self._oracle.variable_register)]
        # sort bitstrings by their probabilities
        linear.sort(key=lambda x: x[1], reverse=True)

        # construct matrix
        equations = []
        for k, _ in linear:
            equations.append([int(c) for c in k])
        y = Matrix(equations)

        # perform gaussian elimination
        y_transformed = y.rref(iszerofunc=lambda x: x % 2 == 0)

        def mod(x, modulus):
            numer, denom = x.as_numer_denom()
            return numer * mod_inverse(denom, modulus) % modulus
        y_new = y_transformed[0].applyfunc(lambda x: mod(x, 2))

        # determine hidden string from final matrix
        rows, _ = y_new.shape
        hidden = [0] * len(self._oracle.variable_register)
        for r in range(rows):
            yi = [i for i, v in enumerate(list(y_new[r, :])) if v == 1]
            if len(yi) == 2:
                hidden[yi[0]] = '1'
                hidden[yi[1]] = '1'
        return "".join(str(x) for x in hidden)[::-1]

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
            measurements = {
                np.binary_repr(idx, width=len(self._oracle.variable_register)):
                    abs(variable_register_density_matrix_diag[idx]) ** 2
                for idx in range(len(variable_register_density_matrix_diag))
                if not variable_register_density_matrix_diag[idx] == 0
            }
        else:
            qc = self.construct_circuit(measurement=True)
            measurements = self._quantum_instance.execute(qc).get_counts(qc)

        self._ret['result'] = self._interpret_measurement(measurements)
        return self._ret
