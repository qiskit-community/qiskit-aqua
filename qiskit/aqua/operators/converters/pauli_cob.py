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

from .. import OpPrimitive, OpComposition, H, S, I

logger = logging.getLogger(__name__)


class PauliChangeOfBasis():
    """ Converter for changing Paulis into other bases. By default, Pauli {Z,I}^n is used as the destination basis.
    Meaning, if a Pauli containing X or Y terms is passed in, which cannot be sampled or evolved natively on Quantum
    hardware, the Pauli can be replaced by a composition of a change of basis circuit and a Pauli composed of only Z
    and I terms, which can be evolved or sampled natively on gate-based Quantum hardware. """

    def __init__(self, destination_basis=None, traverse=True):
        """ Args:
            destination_basis(Pauli): The Pauli into the basis of which the operators will be converted. If None is
            specified, the destination basis will be the {I,Z}^n basis requiring only single qubit rotations.
            travers(bool): If true and the operator passed into convert is an OpVec, traverse the OpVec,
            applying the conversion to every applicable operator within the oplist.
        """
        if destination_basis is not None and isinstance(destination_basis, OpPrimitive):
            destination_basis = destination_basis.primitive
        if destination_basis is not None and not isinstance(destination_basis, Pauli):
            raise TypeError('PauliChangeOfBasis can only convert into Pauli bases, '
                            'not {}.'.format(type(destination_basis)))
        self._destination = destination_basis
        self._traverse = traverse

    # TODO see whether we should make this performant by handling OpVecs of Paulis later.
    def convert(self, operator):
        """ Given an Operator with Paulis, converts each Pauli into the basis specified by self._destination. More
        specifically, each Pauli p will be replaced by the composition of a Change-of-basis Clifford c with the
        destination Pauli d, such that p == c·d·c†, up to global phase. """

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
        return OpComposition([new_pauli, cob_instruction], coeff=coeff)

    def get_cob_circuit(self, pauli):
        # If pauli is an OpPrimitive, extract the Pauli
        if hasattr(pauli, 'primitive') and isinstance(pauli.primitive, Pauli):
            pauli = pauli.primitive

        # If no destination specified, assume nearest Pauli in {Z,I}^n basis, the standard CoB for expectation
        pauli_ones = np.logical_or(pauli.x, pauli.z)
        destination = self._destination or Pauli(z=pauli_ones, x=[False]*len(pauli.z))

        # TODO be smarter about connectivity and actual distance between pauli and destination
        # TODO be smarter in general

        kronall = partial(reduce, lambda x, y: x.kron(y))

        # Construct single-qubit changes to {Z, I)^n
        y_to_x_pauli = kronall([S if has_y else I for has_y in reversed(np.logical_and(pauli.x, pauli.z))]).adjoint()
        # Note, underlying Pauli bits are in Qiskit endian-ness!!
        x_to_z_pauli = kronall([H if has_x else I for has_x in reversed(pauli.x)])
        cob_instruction = x_to_z_pauli.compose(y_to_x_pauli)

        # Construct CNOT chain, assuming full connectivity...
        destination_ones = np.logical_or(destination.x, destination.z)
        lowest_one_dest = min(destination_ones * range(len(pauli.z)))

        non_equal_z_bits = np.logical_xor(pauli_ones, destination_ones)
        if any(non_equal_z_bits):
            cnots = QuantumCircuit(len(pauli.z))
            # Note: Reversing Pauli bit endian-ness!
            for i, val in enumerate(reversed(non_equal_z_bits)):
                if val:
                    cnots.cx(i, lowest_one_dest)
            cnot_op = OpPrimitive(cnots.to_instruction())
            cob_instruction = cnot_op.compose(cob_instruction)

        if any(destination.x):
            # Construct single-qubit changes from {Z, I)^n
            z_to_x_dest = kronall([H if has_x else I for has_x in reversed(destination.x)]).adjoint()
            x_to_y_dest = kronall([S if has_y else I for has_y in reversed(np.logical_and(destination.x,
                                                                                          destination.z))])
            cob_instruction = x_to_y_dest.compose(z_to_x_dest).compose(cob_instruction)
            # cob_instruction = cob_instruction.compose(z_to_x_dest).compose(x_to_y_dest)

        return cob_instruction, OpPrimitive(destination)
