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

"""The Multivariate Variational Distribution."""

import warnings
from typing import Optional, List, Union
import numpy as np

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.aqua.components.variational_forms import VariationalForm
from .multivariate_distribution import MultivariateDistribution

# pylint: disable=invalid-name


class MultivariateVariationalDistribution(MultivariateDistribution):
    """The Multivariate Variational Distribution."""

    def __init__(self,
                 num_qubits: Union[List[int], np.ndarray],
                 var_form: Union[QuantumCircuit, VariationalForm],
                 params: Union[List[float], np.ndarray],
                 low: Optional[Union[List[float], np.ndarray]] = None,
                 high: Optional[Union[List[float], np.ndarray]] = None) -> None:
        """
        Args:
            num_qubits:  List with the number of qubits per dimension
            var_form: Variational form
            params: Parameters for variational form
            low: List with the lower bounds per dimension, set to 0 for each dimension if None
            high:  List with the upper bounds per dimension, set to 1 for each dimension if None
        """
        if low is None:
            low = np.zeros(len(num_qubits))
        if high is None:
            high = np.ones(len(num_qubits))
        self._num_qubits = num_qubits
        self._var_form = var_form
        # fix the order of the parameters in the circuit
        if isinstance(self._var_form, QuantumCircuit):
            self._var_form_params = sorted(self._var_form.parameters, key=lambda p: p.name)
        else:
            warnings.warn('The VariationalForm type is deprecated as argument of the '
                          'MultivariateVariationalDistribution as of 0.7.0 and will be removed no '
                          'earlier than 3 months after the release. You should pass an object '
                          'of type QuantumCircuit instead (see qiskit.circuit.library for a '
                          'collection of suitable objects).',
                          DeprecationWarning, stacklevel=2)

        self.params = params
        probabilities = np.zeros(2 ** sum(num_qubits))
        super().__init__(num_qubits, probabilities, low, high)
        self._var_form = var_form
        self.params = params

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
            quantum_instance (QuantumInstance): Quantum Instance
        """
        q_ = QuantumRegister(self._num_qubits, name='q')
        qc_ = QuantumCircuit(q_)
        if isinstance(self._var_form, QuantumCircuit):
            param_dict = dict(zip(self._var_form_params, self.params))
            circuit_var_form = self._var_form.assign_parameters(param_dict)
        else:
            circuit_var_form = self._var_form.construct_circuit(self.params)

        qc_.append(circuit_var_form.to_instruction(), qc_.qubits)

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
