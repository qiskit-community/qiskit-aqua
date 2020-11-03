# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The raw feature vector circuit."""

from typing import Set, List
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import ParameterVector, ParameterExpression, Gate


class RawFeatureVector(QuantumCircuit):
    """The raw feature vector circuit."""

    def __init__(self, feature_dimension: int) -> None:
        """
        Args:
            feature_dimension: The feature dimension and number of qubits.

        Raises:
            ValueError: -
        """
        num_qubits = np.log2(feature_dimension)
        if int(num_qubits) != num_qubits:
            raise ValueError('feature_dimension must be a power of 2!')

        qr = QuantumRegister(int(num_qubits), 'q')
        super().__init__(qr, name='Raw')

        self._parameters = list(ParameterVector('p', length=feature_dimension))

        # get a gate that acts as placeholder
        placeholder = Gate('Raw', self.num_qubits, self._parameters[:], label='Raw')
        self.append(placeholder, self.qubits)

    @property
    def parameters(self) -> Set[ParameterExpression]:
        """Return the free parameters in the RawFeatureVector.

        Returns:
            A set of the free parameters.
        """
        return set(self.ordered_parameters)

    @property
    def ordered_parameters(self) -> List[ParameterExpression]:
        """Return the free parameters in the RawFeatureVector.

        Returns:
            A list of the free parameters.
        """
        return list(param for param in self._parameters if isinstance(param, ParameterExpression))

    def assign_parameters(self, param_dict, inplace=False):
        """Call the initialize instruction."""
        if not isinstance(param_dict, dict):
            param_dict = dict(zip(self.ordered_parameters, param_dict))

        if inplace:
            dest = self
        else:
            dest = RawFeatureVector(2 ** self.num_qubits)
            dest._parameters = self._parameters.copy()

        # update the parameter list
        for i, param in enumerate(dest._parameters):
            if param in param_dict.keys():
                dest._parameters[i] = param_dict[param]

        # if fully bound call the initialize instruction
        if len(dest.parameters) == 0:
            dest._data = []  # wipe the current data
            parameters = dest._parameters / np.linalg.norm(dest._parameters)
            dest.initialize(parameters, dest.qubits)  # pylint: ignore=no-member

        # else update the placeholder
        else:
            dest._data[0][0].params = dest._parameters

        if not inplace:
            return dest
