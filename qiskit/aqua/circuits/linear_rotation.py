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

"""Linearly-controlled X, Y or Z rotation."""

import warnings

from qiskit.circuit.library.arithmetic import LinearPauliRotations
from qiskit.aqua.utils import CircuitFactory


class LinearRotation(CircuitFactory):
    r"""*DEPRECATED.* Linearly-controlled X, Y or Z rotation.

    .. deprecated:: 0.7.0
       Use Terra's qiskit.circuit.library.LinearPauliRotations instead.

    For a register of state qubits \|x> and a target qubit \|0> this operator acts as:

        \|x>\|0> --> \|x>( cos(slope * x + offset)\|0> + sin(slope * x + offset)\|1> )

    """

    def __init__(self, slope, offset, num_state_qubits, basis='Y', i_state=None, i_target=None):
        """
        Args:
            slope (float): slope of the controlled rotation
            offset (float): offset of the controlled rotation
            num_state_qubits (int): number of qubits representing the state
            basis (str): type of Pauli rotation ('X', 'Y', 'Z')
            i_state (Optional(Union(list, numpy.ndarray))): indices of the state qubits
                (least significant to most significant)
            i_target (Optional(int)): index of target qubit

        Raises:
            ValueError: invalid input
        """
        warnings.warn('The qiskit.aqua.circuits.LinearRotation object is deprecated and will be '
                      'removed no earlier than 3 months after the 0.7.0 release of Qiskit Aqua. '
                      'You should use qiskit.circuit.library.LinearPauliRotations instead.',
                      DeprecationWarning, stacklevel=2)

        super().__init__(num_state_qubits + 1)

        # store the circuit
        self._linear_rotation_circuit = LinearPauliRotations(num_state_qubits=num_state_qubits,
                                                             slope=slope,
                                                             offset=offset,
                                                             basis=basis)

        # store parameters
        self.num_control_qubits = num_state_qubits
        self.slope = slope
        self.offset = offset
        self.basis = basis

        if self.basis not in ['X', 'Y', 'Z']:
            raise ValueError('Basis must be X, Y or Z')

        self.i_state = None
        if i_state is not None:
            self.i_state = i_state
        else:
            self.i_state = range(num_state_qubits)

        self.i_target = None
        if i_target is not None:
            self.i_target = i_target
        else:
            self.i_target = num_state_qubits

    def build(self, qc, q, q_ancillas=None, params=None):
        instr = self._linear_rotation_circuit.to_instruction()
        qr = [q[i] for i in self.i_state] + [q[self.i_target]]
        qc.append(instr, qr)
