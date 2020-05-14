# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The Univariate Variational Distribution."""

import warnings
from typing import Union, List
import numpy as np

from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.aqua.components.variational_forms import VariationalForm
from qiskit.aqua.utils.validation import validate_min
from .univariate_distribution import UnivariateDistribution


class UnivariateVariationalDistribution(UnivariateDistribution):
    """The Univariate Variational Distribution."""

    def __init__(self,
                 num_qubits: int,
                 var_form: Union[QuantumCircuit, VariationalForm],
                 params: Union[List[float], np.ndarray],
                 low: float = 0,
                 high: float = 1) -> None:
        """
        Args:
            num_qubits: Number of qubits
            var_form: Variational form
            params: Parameters for variational form
            low: Lower bound
            high: Upper bound
        """
        validate_min('num_qubits', num_qubits, 1)
        self._num_qubits = num_qubits
        self._var_form = var_form

        # fix the order of the parameters in the circuit
        if isinstance(self._var_form, QuantumCircuit):
            self._var_form_params = sorted(self._var_form.parameters, key=lambda p: p.name)
        else:
            warnings.warn('The VariationalForm type is deprecated as argument of the '
                          'UnivariateVariationalDistribution as of 0.7.0 and will be removed no '
                          'earlier than 3 months after the release. You should pass an object '
                          'of type QuantumCircuit instead (see qiskit.circuit.library for a '
                          'collection of suitable objects).',
                          DeprecationWarning, stacklevel=2)

        self.params = params
        if isinstance(num_qubits, int):
            probabilities = np.zeros(2 ** num_qubits)
        elif isinstance(num_qubits, float):
            probabilities = np.zeros(2 ** int(num_qubits))
        else:
            probabilities = np.zeros(2 ** sum(num_qubits))
        super().__init__(num_qubits, probabilities, low, high)

    def build(self, qc, q, q_ancillas=None, params=None):
        if isinstance(self._var_form, QuantumCircuit):
            param_dict = dict(zip(self._var_form_params, self.params))
            circuit_var_form = self._var_form.assign_parameters(param_dict)
        else:
            circuit_var_form = self._var_form.construct_circuit(self.params)

        qc.append(circuit_var_form.to_instruction(), q)

    def set_probabilities(self, quantum_instance):
        """Set Probabilities

        Args:
            quantum_instance (QuantumInstance): Quantum instance
        """
        if isinstance(self._var_form, QuantumCircuit):
            param_dict = dict(zip(self._var_form_params, self.params))
            qc_ = self._var_form.assign_parameters(param_dict)
        else:
            qc_ = self._var_form.construct_circuit(self.params)

        # q_ = QuantumRegister(self._num_qubits)
        # qc_ = QuantumCircuit(q_)
        # self.build(qc_, None)

        if quantum_instance.is_statevector:
            pass
        else:
            c__ = ClassicalRegister(self._num_qubits, name='c')
            qc_.add_register(c__)
            qc_.measure(qc_.qregs[0], c__)
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
