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

"""The variational form based initial state"""

from typing import Union, List, Dict
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.aqua import AquaError
from qiskit.aqua.components.variational_forms import VariationalForm


class VarFormBased:
    """The variational form based initial state.

    This can been useful, say for example, if you have been doing experiments using a
    :class:`~qiskit.aqua.components.variational_forms.VariationalForm` and have parameters for
    a state of interest of that form. Using this class it can then be turned into an initial state
    for use elsewhere.

    As an example this `notebook
    <https://github.com/Qiskit/qiskit-community-tutorials/blob/master/aqua/vqe2iqpe.ipynb>`__
    shows where the variational form's state, from a :class:`~qiskit.aqua.algorithms.VQE` run,
    is then used as an initial state for :class:`~qiskit.aqua.algorithms.IQPE` by using this
    class.
    """

    def __init__(self,
                 var_form: Union[VariationalForm, QuantumCircuit],
                 params: Union[List[float], np.ndarray, Dict[Parameter, float]]) -> None:
        """
        Args:
            var_form: The variational form.
            params: Parameters for the variational form.
        Raises:
            ValueError: Invalid input
        """
        super().__init__()
        if not var_form.num_parameters == len(params):
            raise ValueError('Incompatible parameters provided.')
        self._var_form = var_form
        self._var_form_params = params

    def construct_circuit(self, mode='circuit', register=None):
        """
        Construct the statevector of desired initial state.

        Args:
            mode (string): `vector` or `circuit`. The `vector` mode produces the vector.
                            While the `circuit` constructs the quantum circuit corresponding that
                            vector.
            register (QuantumRegister): qubits for circuit construction.

        Returns:
            QuantumCircuit or numpy.ndarray: statevector.

        Raises:
            RuntimeError: invalid input for mode
            AquaError: when mode is not 'vector' or 'circuit'.
        """
        if mode == 'vector':
            raise RuntimeError('Initial state based on variational '
                               'form does not support vector mode.')
        if mode == 'circuit':
            if isinstance(self._var_form, VariationalForm):
                return self._var_form.construct_circuit(self._var_form_params, q=register)
            return self._var_form.assign_parameters(self._var_form_params)

        else:
            raise AquaError('Mode should be either "vector" or "circuit"')
