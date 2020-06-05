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
The Deutsch-Jozsa algorithm.
"""

from typing import Optional, Union, Dict, Any
import logging
import operator
import numpy as np

from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.providers import BaseBackend
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import QuantumAlgorithm
from qiskit.aqua.utils import get_subsystem_density_matrix
from qiskit.aqua.components.oracles import Oracle

logger = logging.getLogger(__name__)

# pylint: disable=invalid-name


class DeutschJozsa(QuantumAlgorithm):
    r"""
    The Deutsch-Jozsa algorithm.

    The Deutsch-Jozsa algorithm was one of the first known quantum algorithms that showed an
    exponential speedup compared to a deterministic (non-probabilistic) classical algorithm,
    given a black box oracle function. The algorithm determines whether the given function
    :math:`f:\{0,1\}^n \rightarrow \{0,1\}` is constant or balanced. A constant function maps all
    inputs to 0 or 1, and a balanced function maps half of its inputs to 0 and the other half to 1.

    Note: the :class:`~qiskit.aqua.components.oracles.TruthTableOracle` facilitates creating
    a constant or balanced function but any oracle can be used as long as the boolean function
    implemented by the oracle indeed satisfies the constraint of being either constant or balanced.
    """

    def __init__(self,
                 oracle: Oracle,
                 quantum_instance: Optional[Union[QuantumInstance, BaseBackend]] = None) -> None:
        """
        Args:
            oracle: The oracle component
            quantum_instance: Quantum Instance or Backend
        """
        super().__init__(quantum_instance)

        self._oracle = oracle
        self._circuit = None
        self._ret = {}  # type: Dict[str, Any]

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
