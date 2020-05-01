# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" The Multivariate Variational Distribution. """

import numpy as np

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.aqua import Pluggable, get_pluggable_class, PluggableType
from .multivariate_distribution import MultivariateDistribution

# pylint: disable=invalid-name


class MultivariateVariationalDistribution(MultivariateDistribution):
    """
    The Multivariate Variational Distribution.
    """
    CONFIGURATION = {
        'name': 'MultivariateVariationalDistribution',
        'description': 'Multivariate Variational Distribution',
        'input_schema': {
            '$schema': 'http://json-schema.org/draft-07/schema#',
            'id': 'MultivariateVariationalDistribution_schema',
            'type': 'object',
            'properties': {
                'num_qubits': {
                    'type': 'array',
                    "items": {
                        "type": "number"
                    }
                },

                'params': {
                    'type': 'array',
                    "items": {
                        "type": "number"
                    }
                },
                'low': {
                    'type': ['array', 'null'],
                    "items": {
                        "type": "number"
                    },
                    'default': None
                },
                'high': {
                    'type': ['array', 'null'],
                    "items": {
                        "type": "number"
                    },
                    'default': None
                },
            },
            'additionalProperties': False
        },
        'depends': [
            {
                'pluggable_type': 'variational_form',
                'default': {
                    'name': 'RY'
                }
            }

        ],
    }

    def __init__(self, num_qubits, var_form, params, low=None, high=None):
        if low is None:
            low = np.zeros(len(num_qubits))
        if high is None:
            high = np.ones(len(num_qubits))
        self._num_qubits = num_qubits
        self._var_form = var_form
        self.params = params
        probabilities = np.zeros(2 ** sum(num_qubits))
        super().__init__(num_qubits, probabilities, low, high)
        self._var_form = var_form
        self.params = params

    @classmethod
    def init_params(cls, params):
        """
        Initialize via parameters dictionary.
        Args:
            params (dict): parameters dictionary
        Returns:
            MultivariateVariationalDistribution: An object instance of this class
        """

        multi_var_params_params = params.get(Pluggable.SECTION_KEY_UNIVARIATE_DIST)
        num_qubits = multi_var_params_params.get('num_qubits')
        params = multi_var_params_params.get('params')
        low = multi_var_params_params.get('low')
        high = multi_var_params_params.get('high')

        var_form_params = params.get(Pluggable.SECTION_KEY_VAR_FORM)
        var_form = get_pluggable_class(PluggableType.VARIATIONAL_FORM,
                                       var_form_params['name']).init_params(params)

        return cls(num_qubits, var_form, params, low, high)

    def build(self, qc, q, q_ancillas=None, params=None):
        circuit_var_form = self._var_form.construct_circuit(self.params)
        qc.append(circuit_var_form.to_instruction(), q)

    def set_probabilities(self, quantum_instance):
        """
        Set Probabilities
        Args:
            quantum_instance (QuantumInstance): Quantum Instance
        """
        q_ = QuantumRegister(self._num_qubits, name='q')
        qc_ = QuantumCircuit(q_)
        circuit_var_form = self._var_form.construct_circuit(self.params, q_)
        qc_ += circuit_var_form

        if quantum_instance.is_statevector:
            pass
        else:
            c_ = ClassicalRegister(self._num_qubits, name='c')
            qc_.add_register(c_)
            qc_.measure(q_, c_)
        result = quantum_instance.execute(qc_)
        if quantum_instance.is_statevector:
            result = result.get_statevector(qc_)
            values = np.multiply(result, np.conj(result))
            values = list(values.real)
        else:
            result = result.get_counts(qc_)
            keys = list(result)
            values = list(result.values())
            values = [float(v) / np.sum(values) for v in values]
            values = [x for _, x in sorted(zip(keys, values))]

        probabilities = values
        self._probabilities = np.array(probabilities)
