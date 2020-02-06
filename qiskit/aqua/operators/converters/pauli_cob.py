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

""" Expectation Algorithm Base """

import logging
import numpy as np
from functools import partial, reduce

from qiskit.quantum_info import Pauli
from qiskit import QuantumCircuit

from .. import OperatorBase, OpPrimitive, OpComposition, H, S, CX

logger = logging.getLogger(__name__)


class PauliChangeOfBasis():
    """ Converter for changing Paulis into other bases. By default, Pauli {Z,I}^n is used as the destination basis.
    Meaning, if a Pauli containing X or Y terms is passed in, which cannot be sampled or evolved natively on Quantum
    hardware, the Pauli can be replaced by a composition of a change of basis circuit and a Pauli composed of only Z
    and I terms, which can be evolved or sampled natively on gate-based Quantum hardware. """

    def __init__(self, destination_basis=None, measure=True, traverse=True, change_back=True):
        """ Args:
            destination_basis(Pauli): The Pauli into the basis of which the operators will be converted. If None is
            specified, the destination basis will be the {I,Z}^n basis requiring only single qubit rotations.
        """
        if destination_basis is not None and not isinstance(destination_basis, Pauli):
            raise TypeError('PauliChangeOfBasis can only convert into Pauli bases, '
                            'not {}.'.format(type(destination_basis)))
        self._destination = destination_basis
        self._traverse = traverse
        self._change_back = change_back

    # TODO see whether we should make this performant by handling OpVecs of Paulis later.
    def convert(self, operator):
        if isinstance(operator, Pauli):
            pauli = operator
            coeff = 1.0
        elif hasattr(operator, 'primitive') and isinstance(operator.primitive, Pauli):
            pauli = operator.primitive
            coeff = operator.coeff
        elif isinstance(operator, OpVec) and self._traverse and 'Pauli' in operator.get_primitives():
            return operator.traverse(self.convert)
        else:
            raise TypeError('PauliChangeOfBasis can only accept OperatorBase objects or '
                            'Paulis, not {}'.format(type(operator)))

        cob_instruction, new_pauli = self.get_cob_circuit(pauli)
        return OpComposition([cob_instruction, new_pauli], coeff=coeff)

    def get_cob_circuit(self, pauli):
        # If no destination specified, assume nearest Pauli in {Z,I}^n basis
        destination = self._destination or Pauli(z=pauli.z)
        # TODO be smarter about connectivity
        # TODO be smarter in general
        kronall = partial(reduce, lambda x, y: x.kron(y))

        # Construct single-qubit changes to {Z, I)^n
        y_to_x_pauli = kronall([S.adjoint() if has_y else I for has_y in np.logical_and(pauli.x, pauli.z)])
        x_to_z_pauli = kronall([H if has_x else I for has_x in pauli.x])

        # Construct CNOT chain, assuming full connectivity...
        pauli_ones = np.logical_or(pauli.x, pauli.z)
        destination_ones = np.logical_or(destination.x, destination.z)
        lowest_one_dest = min(range(destination_ones * len(pauli.z)))
        cnots = QuantumCircuit(len(pauli.z))
        for i, val in enumerate(np.logical_xor(pauli_ones, destination_ones)):
            if val:
                cnots.cx(i, lowest_one_dest)
        cnot_op = OpPrimitive(cnots.to_instruction())

        # Construct single-qubit changes from {Z, I)^n
        z_to_x_dest = kronall([H if has_x else I for has_x in destination.x]).adjoint()
        x_to_y_dest = kronall([S if has_y else I for has_y in np.logical_and(destination.x, destination.z)]).adjoint()

        cob_instruction = y_to_x_pauli.compose(x_to_z_pauli).compose(cnot_op).compose(z_to_x_dest).compose(x_to_y_dest)

        return cob_instruction, destination