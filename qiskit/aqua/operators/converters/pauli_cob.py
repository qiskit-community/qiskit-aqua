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

from .. import OpPrimitive, OpPauli, OpComposition, OpVec, StateFn, H, S, I
from . import ConverterBase

logger = logging.getLogger(__name__)


class PauliChangeOfBasis(ConverterBase):
    """ Converter for changing Paulis into other bases. By default, Pauli {Z,I}^n is used as the destination basis.
    Meaning, if a Pauli containing X or Y terms is passed in, which cannot be sampled or evolved natively on Quantum
    hardware, the Pauli can be replaced by a composition of a change of basis circuit and a Pauli composed of only Z
    and I terms, which can be evolved or sampled natively on gate-based Quantum hardware. """

    def __init__(self, destination_basis=None, traverse=True, replacement_fn=None):
        """ Args:
            destination_basis(Pauli): The Pauli into the basis of which the operators will be converted. If None is
            specified, the destination basis will be the {I,Z}^n basis requiring only single qubit rotations.
            travers(bool): If true and the operator passed into convert is an OpVec, traverse the OpVec,
            applying the conversion to every applicable operator within the oplist.
            replacement_fn(callable): A function specifying what to do with the CoB instruction and destination
            Pauli when converting an Operator and replacing converted values. By default, this will be
                1) For StateFns (or Measurements): replacing the StateFn with OpComposition(StateFn(d), c) where c
                is the conversion circuit and d is the destination Pauli, so the overall beginning and
                ending operators are equivalent.
                2) For non-StateFn Operators: replacing the origin p with c·d·c†, where c is the conversion circuit
                and d is the destination, so the overall beginning and ending operators are equivalent.
        """
        if destination_basis is not None:
            self.destination = destination_basis
        else:
            self._destination = None
        self._traverse = traverse
        self._replacement_fn = replacement_fn

    @property
    def destination(self):
        return self._destination

    @destination.setter
    def destination(self, dest):
        if isinstance(dest, Pauli):
            dest = OpPauli(dest)

        if not isinstance(dest, OpPauli):
            raise TypeError('PauliChangeOfBasis can only convert into Pauli bases, '
                            'not {}.'.format(type(dest)))
        else:
            self._destination = dest

    # TODO see whether we should make this performant by handling OpVecs of Paulis later.
    def convert(self, operator):
        """ Given an Operator with Paulis, converts each Pauli into the basis specified by self._destination. More
        specifically, each Pauli p will be replaced by the composition of a Change-of-basis Clifford c with the
        destination Pauli d and c†, such that p == c·d·c†, up to global phase. """

        if isinstance(operator, (Pauli, OpPrimitive)):
            origin_pauli = operator
            # Don't need to set coeff for OpPrimitive because converter below will set it in dest_pauli if available
            coeff = 1.0
        elif isinstance(operator, StateFn) and 'Pauli' in operator.get_primitives():
            # If the StateFn/Meas only contains a Pauli, use it directly.
            if isinstance(operator.primitive, OpPrimitive):
                origin_pauli = operator.primitive
                coeff = operator.coeff
            # TODO make a cononical "distribute" or graph swap as method in OperatorBase
            elif operator.primitive.distributive:
                sf_list = [StateFn(op, is_measurement=operator.is_measurement) for op in operator.primitive.oplist]
                opvec_of_statefns = operator.primitive.__class__(oplist=sf_list, coeff=operator.coeff)
                return opvec_of_statefns.traverse(self.convert)

        # TODO allow parameterized OpVec to be returned to save circuit copying.
        elif isinstance(operator, OpVec) and self._traverse and 'Pauli' in operator.get_primitives():
            # If opvec is abelian we can find a single post-rotation circuit for the whole set. For now,
            # assume operator can only be abelian if all elements are Paulis (enforced in AbelianGrouper).
            if operator.abelian:
                origin_z = reduce(np.logical_or, [p_op.primitive.z for p_op in operator.oplist])
                origin_x = reduce(np.logical_or, [p_op.primitive.x for p_op in operator.oplist])
                origin_pauli = Pauli(x=origin_x, z=origin_z)
            else:
                return operator.traverse(self.convert)
        else:
            raise TypeError('PauliChangeOfBasis can only accept OperatorBase objects or '
                            'Paulis, not {}'.format(type(operator)))

        cob_instr_op, dest_pauli_op = self.get_cob_circuit(origin_pauli)

        if isinstance(operator, OpVec) and operator.abelian:
            diag_ops = [self.get_diagonal_pauli_op(op) for op in operator.oplist]
            dest_pauli_op = operator.__class__(diag_ops, coeff=operator.coeff, abelian=True)

        if self._replacement_fn:
            return self._replacement_fn(cob_instr_op, dest_pauli_op)
        elif isinstance(operator, StateFn):
            new_sf = OpComposition([cob_instr_op.adjoint(), StateFn(dest_pauli_op)], coeff=coeff)
            return new_sf.adjoint() if operator.is_measurement else new_sf
        else:
            return OpComposition([cob_instr_op.adjoint(), dest_pauli_op, cob_instr_op], coeff=coeff)

    def get_diagonal_pauli_op(self, pauli_op):
        return OpPauli(Pauli(z=np.logical_or(pauli_op.primitive.z, pauli_op.primitive.x),
                             x=[False] * pauli_op.num_qubits),
                       coeff=pauli_op.coeff)

    def get_diagonalizing_clifford(self, pauli):
        """ Construct single-qubit rotations to {Z, I)^n
         Note, underlying Pauli bits are in Qiskit endian-ness!! """
        if isinstance(pauli, OpPauli):
            pauli = pauli.primitive

        kronall = partial(reduce, lambda x, y: x.kron(y))

        y_to_x_origin = kronall([S if has_y else I for has_y in reversed(np.logical_and(pauli.x, pauli.z))]).adjoint()
        x_to_z_origin = kronall([H if has_x else I for has_x in reversed(pauli.x)])
        return x_to_z_origin.compose(y_to_x_origin)

    def pad_paulis_to_equal_length(self, pauli_op1, pauli_op2):
        num_qubits = max(pauli_op1.num_qubits, pauli_op2.num_qubits)
        pauli_1, pauli_2 = pauli_op1.primitive, pauli_op2.primitive

        if not len(pauli_1.z) == num_qubits:
            missing_qubits = num_qubits - len(pauli_1.z)
            pauli_1 = Pauli(z=pauli_1.z.tolist() + ([False] * missing_qubits),
                            x=pauli_1.x.tolist() + ([False] * missing_qubits))
        if not len(pauli_2.z) == num_qubits:
            missing_qubits = num_qubits - len(pauli_2.z)
            pauli_2 = Pauli(z=pauli_2.z.tolist() + ([False] * missing_qubits),
                            x=pauli_2.x.tolist() + ([False] * missing_qubits))

        return OpPauli(pauli_1, coeff=pauli_op1.coeff), OpPauli(pauli_2, coeff=pauli_op2.coeff)

    # TODO
    def construct_cnot_chain(self, diag_pauli_op1, diag_pauli_op2):
        # TODO be smarter about connectivity and actual distance between pauli and destination
        # TODO be smarter in general

        pauli_1 = diag_pauli_op1.primitive if isinstance(diag_pauli_op1, OpPauli) else diag_pauli_op1
        pauli_2 = diag_pauli_op2.primitive if isinstance(diag_pauli_op2, OpPauli) else diag_pauli_op2
        origin_sig_bits = np.logical_or(pauli_1.z, pauli_1.x)
        destination_sig_bits = np.logical_or(pauli_2.z, pauli_2.x)
        # TODO maybe just raise error if not equal
        num_qubits = max(len(pauli_1.z), len(pauli_2.z))

        sig_equal_sig_bits = np.logical_and(origin_sig_bits, destination_sig_bits)
        non_equal_sig_bits = np.logical_not(origin_sig_bits == destination_sig_bits)
        # Equivalent to np.logical_xor(origin_sig_bits, destination_sig_bits)

        if not any(non_equal_sig_bits):
            return I^num_qubits

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
            origin_anchor_bit = min(np.extract(sig_equal_sig_bits, np.arange(num_qubits)))
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

        # TODO seems like we don't need this
        # Step 5)
        # if not len(sig_in_origin_only_indices) % 2 == len(sig_in_dest_only_indices) % 2:
        #     cnots.x(dest_anchor_bit)

        # Need to do this or a Terra bug sometimes flips cnots. No time to investigate.
        cnots.i(0)

        # Step 6)
        for i in sig_in_dest_only_indices:
            if not i == dest_anchor_bit:
                cnots.cx(i, dest_anchor_bit)

        # TODO seems like we don't need this
        # Step 7)
        # if not len(sig_in_origin_only_indices) % 2 == len(sig_in_dest_only_indices) % 2:
        #     cnots.x(dest_anchor_bit)

        return OpPrimitive(cnots.to_instruction())

    # TODO
    def compute_shortest_cnot_path(self, ablian_op_vec):
        pass

    # TODO change to only accept OpPrimitive Pauli.
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
        if isinstance(origin, Pauli):
            origin = OpPauli(origin)

        if not isinstance(origin, OpPauli):
            raise TypeError('PauliCoB can only convert Pauli-based OpPrimitives, not {}'.format(type(
                OpPrimitive.primitive)))

        # If no destination specified, assume nearest Pauli in {Z,I}^n basis, the standard CoB for expectation
        destination = self.destination or self.get_diagonal_pauli_op(origin)

        # Pad origin or destination if either are not as long as the other
        origin, destination = self.pad_paulis_to_equal_length(origin, destination)

        origin_sig_bits = np.logical_or(origin.primitive.x, origin.primitive.z)
        destination_sig_bits = np.logical_or(destination.primitive.x, destination.primitive.z)
        if not any(origin_sig_bits) or not any(destination_sig_bits):
            if not (any(origin_sig_bits) or any(destination_sig_bits)):
                # Both all Identity, just return Identities
                return I^origin.num_qubits, destination
            else:
                # One is Identity, one is not
                raise ValueError('Cannot change to or from a fully Identity Pauli.')

        # Steps 1 and 2
        cob_instruction = self.get_diagonalizing_clifford(origin)

        # Construct CNOT chain, assuming full connectivity... - Steps 3)-7)
        cob_instruction = self.construct_cnot_chain(origin, destination).compose(cob_instruction)

        # Step 8 and 9
        dest_diagonlizing_clifford = self.get_diagonalizing_clifford(destination).adjoint()
        cob_instruction = dest_diagonlizing_clifford.compose(cob_instruction)

        return cob_instruction, destination
