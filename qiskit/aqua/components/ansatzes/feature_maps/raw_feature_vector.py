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

"""The raw feature vector Ansatz. Encodes given data in the qubit amplitudes."""

from typing import List, Optional
import logging
import numpy as np

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import ParameterVector, Instruction, ParameterExpression

from qiskit.aqua.aqua_error import AquaError
from qiskit.aqua.utils.arithmetic import next_power_of_2_base
from qiskit.aqua.circuits import StateVectorCircuit
from qiskit.aqua.utils.validation import validate_min
from qiskit.aqua.components.ansatzes import Ansatz

logger = logging.getLogger(__name__)


class RawFeatureVector(Ansatz):
    """
    Using raw feature vector as the initial state vector
    """

    def __init__(self, feature_dimension: int = 2) -> None:
        """Constructor.

        Args:
            feature_dimension: The feature dimension, has a min. value of 1.
        """
        validate_min('feature_dimension', feature_dimension, 1)
        self._feature_dimension = feature_dimension
        self._num_qubits = next_power_of_2_base(feature_dimension)

        x = ParameterVector('x', feature_dimension)
        placeholder = Instruction('x', self._num_qubits, 0, [*x])
        super().__init__(blocks=placeholder, overwrite_block_parameters=False)

    @property
    def support_parameterized_circuit(self):
        """TODO Deprecate.

        Whether it is supported to bind parameters in this circuit.
        """
        return False

    @property
    def feature_dimension(self):
        """Return the feature dimension."""
        return self._feature_dimension

    def to_circuit(self) -> QuantumCircuit:
        """Construct the circuit.

        Returns:
            The circuit representing the feature vector.

        Raises:
            AquaError: If the parameters contain a ParameterExpression.
        """
        x = self.parameters
        if any(isinstance(xi, ParameterExpression) for xi in x):
            raise AquaError('Cannot construct the RawFeatureVector on a ParameterExpression.')

        state_vector = np.pad(x, (0, (1 << self.num_qubits) - self.feature_dimension), 'constant')
        self._circuit = StateVectorCircuit(state_vector).construct_circuit()
        return self._circuit

    def construct_circuit(self, x: List[float],   # pylint:disable=arguments-differ
                          qr: Optional[QuantumRegister] = None) -> QuantumCircuit:
        """Construct the circuit for the given data ``x``.

        Args:
            x: 1-D to-be-encoded data.
            qr: The QuantumRegister object for the circuit, if None,
                generate new registers with name q.

        Returns:
            A quantum circuit encoding the data ``x``.

        Raises:
            ValueError: Dimension of ``x`` is not equal to the feature dimension.
        """
        if len(x) != self._feature_dimension:
            raise ValueError("Unexpected feature vector dimension.")

        return super().construct_circuit(params=x, q=qr)
