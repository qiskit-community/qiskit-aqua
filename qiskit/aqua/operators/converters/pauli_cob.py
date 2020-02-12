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

from .. import OpPrimitive, OpComposition, OpVec, H, S, I

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
            self._destination = destination_basis.primitive
        else:
            self._destination = destination_basis
        if self._destination is not None and not isinstance(self._destination, Pauli):
            raise TypeError('PauliChangeOfBasis can only convert into Pauli bases, '
                            'not {}.'.format(type(destination_basis)))
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

    def get_cob_circuit(self, origin):
        """ The goal of this module is to construct a circuit which maps the +1 and -1 eigenvectors of the origin
        pauli to the +1 and -1 eigenvectors of the destination pauli. It does so by
            1) converting any |i+⟩ or |i+⟩ eigenvector bits in the origin to |+⟩ and |-⟩ with S†s, then
            2) converting any |+⟩ or |+⟩ eigenvector bits in the converted origin to |0⟩ and |1⟩ with Hs, then
            3) writing the parity of the significant (Z-measured, rather than I) bits in the origin to a single
            "origin anchor bit," using cnots, which will hold the parity of these bits,
            4) swapping the parity of the pauli anchor bit into a destination anchor bit using a swap gate (only if
            they are different, if there are any bits which are significant in both origin and dest, we set both
            anchors to one of these bits to avoid a swap).
            5) flipping the state (parity) of the destination anchor if the parity of the number of pauli significant
            bits is different from the parity of the number of destination significant bits (to be flipped back in
            step 7)
            6) writing the parity of the destination anchor bit into the other significant bits of the destination,
            7) flipping back the parity of the destination anchor if we flipped it in step 5)
            8) converting the |0⟩ and |1⟩ significant eigenvector bits to |+⟩ and |-⟩ eigenvector bits in the
            destination where the destination demands it (e.g. pauli.x == true for a bit), using Hs
            8) converting the |+⟩ and |-⟩ significant eigenvector bits to |i+⟩ and |i-⟩ eigenvector bits in the
            destination where the destination demands it (e.g. pauli.x == true and pauli.z == true for a bit), using Ss
        """

        # If pauli is an OpPrimitive, extract the Pauli
        if hasattr(origin, 'primitive') and isinstance(origin.primitive, Pauli):
            origin = origin.primitive

        # If no destination specified, assume nearest Pauli in {Z,I}^n basis, the standard CoB for expectation
        origin_sig_bits = np.logical_or(origin.x, origin.z)
        destination = self._destination or Pauli(z=origin_sig_bits, x=[False]*len(origin.z))
        destination_sig_bits = np.logical_or(destination.x, destination.z)
        num_qubits = max([len(origin.z), len(destination.z)])

        if not any(origin_sig_bits) or not any(destination_sig_bits):
            if not (any(origin_sig_bits) or any(destination_sig_bits)):
                # Both all Identity, just return Identities
                return OpPrimitive(origin), OpPrimitive(destination)
            else:
                # One is Identity, one is not
                raise ValueError('Cannot change to or from a fully Identity Pauli.')

        # TODO be smarter about connectivity and actual distance between pauli and destination
        # TODO be smarter in general

        kronall = partial(reduce, lambda x, y: x.kron(y))

        # Construct single-qubit changes to {Z, I)^n
        # Note, underlying Pauli bits are in Qiskit endian-ness!!
        # Step 1)
        y_to_x_origin = kronall([S if has_y else I for has_y in reversed(np.logical_and(origin.x, origin.z))]).adjoint()
        # Step 2)
        x_to_z_origin = kronall([H if has_x else I for has_x in reversed(origin.x)])
        cob_instruction = x_to_z_origin.compose(y_to_x_origin)

        # Construct CNOT chain, assuming full connectivity... - Steps 3)-7)
        equal_sig_bits = np.logical_and(origin_sig_bits, destination_sig_bits)
        non_equal_sig_bits = np.logical_not(origin_sig_bits == destination_sig_bits)
        # Equivalent to np.logical_xor(origin_sig_bits, destination_sig_bits)

        if any(non_equal_sig_bits):
            # I am deeply sorry for this code, but I don't know another way to do it.
            sig_in_origin_only_indices = np.extract(np.logical_and(non_equal_sig_bits, origin_sig_bits),
                                                    np.arange(num_qubits))
            sig_in_dest_only_indices = np.extract(np.logical_and(non_equal_sig_bits, destination_sig_bits),
                                                  np.arange(num_qubits))

            if len(sig_in_origin_only_indices) and len(sig_in_dest_only_indices):
                origin_anchor_bit = min(sig_in_origin_only_indices)
                dest_anchor_bit = min(sig_in_dest_only_indices)
            else:
                # Set to lowest equal bit
                origin_anchor_bit = min(np.extract(equal_sig_bits, np.arange(num_qubits)))
                dest_anchor_bit = origin_anchor_bit

            cnots = QuantumCircuit(num_qubits)
            # Step 3) Take the indices of bits which are sig_bits in pauli but but not in dest, and cnot them to the
            # pauli anchor.
            for i in sig_in_origin_only_indices:
                if not i == origin_anchor_bit:
                    cnots.cx(i, origin_anchor_bit)

            # Step 4)
            if not origin_anchor_bit == dest_anchor_bit:
                cnots.swap(origin_anchor_bit, dest_anchor_bit)

            # # Step 5)
            # if not len(sig_in_origin_only_indices) % 2 == len(sig_in_dest_only_indices) % 2:
            #     cnots.x(dest_anchor_bit)

            cnots.iden(0)

            # Step 6)
            for i in sig_in_dest_only_indices:
                if not i == dest_anchor_bit:
                    cnots.cx(i, dest_anchor_bit)

            # # Step 7)
            # if not len(sig_in_origin_only_indices) % 2 == len(sig_in_dest_only_indices) % 2:
            #     cnots.x(dest_anchor_bit)

            cnot_op = OpPrimitive(cnots.to_instruction())
            cob_instruction = cnot_op.compose(cob_instruction)

        # Construct single-qubit changes from {Z, I)^n
        if any(destination.x):
            # Step 8)
            z_to_x_dest = kronall([H if has_x else I for has_x in reversed(destination.x)])
            # Step 9)
            x_to_y_dest = kronall([S if has_y else I for has_y in reversed(np.logical_and(destination.x,
                                                                                          destination.z))])
            cob_instruction = x_to_y_dest.compose(z_to_x_dest).compose(cob_instruction)
            # cob_instruction = cob_instruction.compose(z_to_x_dest).compose(x_to_y_dest)

        return cob_instruction, OpPrimitive(destination)
