# -*- coding: utf-8 -*-

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

""" Expectation Algorithm Base """

from typing import Optional, Callable, Union
import logging
from functools import partial, reduce
import numpy as np

from qiskit.quantum_info import Pauli
from qiskit import QuantumCircuit

from ..operator_base import OperatorBase
from ..primitive_operators import PrimitiveOp, PauliOp, CircuitOp
from ..combo_operators import ListOp, ComposedOp
from ..state_functions import StateFn
from ..operator_globals import H, S, I
from .converter_base import ConverterBase

logger = logging.getLogger(__name__)


class PauliBasisChange(ConverterBase):
    """ Converter for changing Paulis into other bases. By default,
    Pauli {Z,I}^n is used as the destination basis.
    Meaning, if a Pauli containing X or Y terms is passed in, which cannot be
    sampled or evolved natively on Quantum
    hardware, the Pauli can be replaced by a composition of a change of basis
    circuit and a Pauli composed of only Z
    and I terms, which can be evolved or sampled natively on gate-based Quantum hardware. """

    def __init__(self,
                 destination_basis: Optional[Union[Pauli, PauliOp]] = None,
                 traverse: bool = True,
                 replacement_fn: Optional[Callable] = None) -> None:
        """ Args:
            destination_basis: The Pauli into the basis of which the operators
            will be converted. If None is
            specified, the destination basis will be the {I,Z}^n basis requiring only
            single qubit rotations.
            traverse: If true and the operator passed into convert is an ListOp,
            traverse the ListOp,
            applying the conversion to every applicable operator within the oplist.
            replacement_fn: A function specifying what to do with the CoB
            instruction and destination
            Pauli when converting an Operator and replacing converted values.
            By default, this will be
                1) For StateFns (or Measurements): replacing the StateFn with
                ComposedOp(StateFn(d), c) where c
                is the conversion circuit and d is the destination Pauli,
                so the overall beginning and ending operators are equivalent.
                2) For non-StateFn Operators: replacing the origin p with c·d·c†,
                where c is the conversion circuit
                and d is the destination, so the overall beginning and ending
                operators are equivalent.
        """
        if destination_basis is not None:
            self.destination = destination_basis
        else:
            self._destination = None
        self._traverse = traverse
        self._replacement_fn = replacement_fn or PauliBasisChange.operator_replacement_fn

    @property
    def destination(self) -> PauliOp:
        """ returns destination """
        return self._destination

    @destination.setter
    def destination(self, dest: Union[Pauli, PauliOp]) -> None:
        if isinstance(dest, Pauli):
            dest = PauliOp(dest)

        if not isinstance(dest, PauliOp):
            raise TypeError('PauliBasisChange can only convert into Pauli bases, '
                            'not {}.'.format(type(dest)))
        self._destination = dest

    # TODO see whether we should make this performant by handling ListOps of Paulis later.
    # pylint: disable=inconsistent-return-statements
    def convert(self, operator: OperatorBase) -> OperatorBase:
        """ Given an Operator with Paulis, converts each Pauli into the basis specified
        by self._destination. More
        specifically, each Pauli p will be replaced by the composition of
        a Change-of-basis Clifford c with the
        destination Pauli d and c†, such that p == c·d·c†, up to global phase. """

        if isinstance(operator, (Pauli, PrimitiveOp)):
            cob_instr_op, dest_pauli_op = self.get_cob_circuit(operator)
            return self._replacement_fn(cob_instr_op, dest_pauli_op)
        if isinstance(operator, StateFn) and 'Pauli' in operator.get_primitives():
            # If the StateFn/Meas only contains a Pauli, use it directly.
            if isinstance(operator.primitive, PrimitiveOp):
                cob_instr_op, dest_pauli_op = self.get_cob_circuit(operator.primitive)
                return self._replacement_fn(cob_instr_op, dest_pauli_op)
            # TODO make a canonical "distribute" or graph swap as method in ListOp?
            elif operator.primitive.distributive:
                if operator.primitive.abelian:
                    origin_pauli = self.get_tpb_pauli(operator.primitive)
                    cob_instr_op, _ = self.get_cob_circuit(origin_pauli)
                    diag_ops = [self.get_diagonal_pauli_op(op) for op in operator.primitive.oplist]
                    dest_pauli_op = operator.primitive.__class__(diag_ops,
                                                                 coeff=operator.coeff, abelian=True)
                    return self._replacement_fn(cob_instr_op, dest_pauli_op)
                else:
                    sf_list = [StateFn(op, is_measurement=operator.is_measurement)
                               for op in operator.primitive.oplist]
                    listop_of_statefns = operator.primitive.__class__(oplist=sf_list,
                                                                      coeff=operator.coeff)
                    return listop_of_statefns.traverse(self.convert)

        # TODO allow parameterized ListOp to be returned to save circuit copying.
        elif isinstance(operator, ListOp) and self._traverse and \
                'Pauli' in operator.get_primitives():
            # If ListOp is abelian we can find a single post-rotation circuit
            # for the whole set. For now,
            # assume operator can only be abelian if all elements are
            # Paulis (enforced in AbelianGrouper).
            if operator.abelian:
                origin_pauli = self.get_tpb_pauli(operator)
                cob_instr_op, _ = self.get_cob_circuit(origin_pauli)
                diag_ops = [self.get_diagonal_pauli_op(op) for op in operator.oplist]
                dest_pauli_op = operator.__class__(diag_ops, coeff=operator.coeff, abelian=True)
                return self._replacement_fn(cob_instr_op, dest_pauli_op)
            else:
                return operator.traverse(self.convert)
        else:
            raise TypeError('PauliBasisChange can only accept OperatorBase objects or '
                            'Paulis, not {}'.format(type(operator)))

    @staticmethod
    def measurement_replacement_fn(cob_instr_op: CircuitOp,
                                   dest_pauli_op: PauliOp) -> OperatorBase:
        """ measurement replacement function """
        return PauliBasisChange.statefn_replacement_fn(cob_instr_op, dest_pauli_op).adjoint()

    @staticmethod
    def statefn_replacement_fn(cob_instr_op: CircuitOp,
                               dest_pauli_op: PauliOp) -> OperatorBase:
        """ state function replacement """
        return ComposedOp([cob_instr_op.adjoint(), StateFn(dest_pauli_op)])

    @staticmethod
    def operator_replacement_fn(cob_instr_op: CircuitOp,
                                dest_pauli_op: PauliOp) -> OperatorBase:
        """ operator replacement """
        return ComposedOp([cob_instr_op.adjoint(), dest_pauli_op, cob_instr_op])

    def get_tpb_pauli(self, op_vec: ListOp) -> Pauli:
        """ get tpb pauli """
        origin_z = reduce(np.logical_or, [p_op.primitive.z for p_op in op_vec.oplist])
        origin_x = reduce(np.logical_or, [p_op.primitive.x for p_op in op_vec.oplist])
        return Pauli(x=origin_x, z=origin_z)

    def get_diagonal_pauli_op(self, pauli_op: PauliOp) -> PauliOp:
        """ get diagonal pauli operation """
        return PauliOp(Pauli(z=np.logical_or(pauli_op.primitive.z, pauli_op.primitive.x),
                             x=[False] * pauli_op.num_qubits),
                       coeff=pauli_op.coeff)

    def get_diagonalizing_clifford(self, pauli: Union[Pauli, PauliOp]) -> OperatorBase:
        """ Construct single-qubit rotations to {Z, I)^n
         Note, underlying Pauli bits are in Qiskit endianness!! """
        if isinstance(pauli, PauliOp):
            pauli = pauli.primitive

        tensorall = partial(reduce, lambda x, y: x.tensor(y))

        # pylint: disable=bad-reversed-sequence
        y_to_x_origin = tensorall([S if has_y else I for has_y in
                                   reversed(np.logical_and(pauli.x, pauli.z))]).adjoint()
        x_to_z_origin = tensorall([H if has_x else I for has_x in
                                   reversed(pauli.x)])
        return x_to_z_origin.compose(y_to_x_origin)

    def pad_paulis_to_equal_length(self,
                                   pauli_op1: PauliOp,
                                   pauli_op2: PauliOp) -> (PauliOp, PauliOp):
        """ pad paulis to equal length """
        num_qubits = max(pauli_op1.num_qubits, pauli_op2.num_qubits)
        pauli_1, pauli_2 = pauli_op1.primitive, pauli_op2.primitive

        # Padding to the end of the Pauli, but remember that Paulis are in reverse endianness.
        if not len(pauli_1.z) == num_qubits:
            missing_qubits = num_qubits - len(pauli_1.z)
            pauli_1 = Pauli(z=([False] * missing_qubits) + pauli_1.z.tolist(),
                            x=([False] * missing_qubits) + pauli_1.x.tolist())
        if not len(pauli_2.z) == num_qubits:
            missing_qubits = num_qubits - len(pauli_2.z)
            pauli_2 = Pauli(z=([False] * missing_qubits) + pauli_2.z.tolist(),
                            x=([False] * missing_qubits) + pauli_2.x.tolist())

        return PauliOp(pauli_1, coeff=pauli_op1.coeff), PauliOp(pauli_2, coeff=pauli_op2.coeff)

    # TODO
    def construct_cnot_chain(self,
                             diag_pauli_op1: PauliOp,
                             diag_pauli_op2: PauliOp) -> PrimitiveOp:
        """ construct cnot chain """
        # TODO be smarter about connectivity and actual distance between pauli and destination
        # TODO be smarter in general

        pauli_1 = diag_pauli_op1.primitive if isinstance(diag_pauli_op1, PauliOp) \
            else diag_pauli_op1
        pauli_2 = diag_pauli_op2.primitive if isinstance(diag_pauli_op2, PauliOp) \
            else diag_pauli_op2
        origin_sig_bits = np.logical_or(pauli_1.z, pauli_1.x)
        destination_sig_bits = np.logical_or(pauli_2.z, pauli_2.x)
        # TODO maybe just raise error if not equal
        num_qubits = max(len(pauli_1.z), len(pauli_2.z))

        sig_equal_sig_bits = np.logical_and(origin_sig_bits, destination_sig_bits)
        non_equal_sig_bits = np.logical_not(origin_sig_bits == destination_sig_bits)
        # Equivalent to np.logical_xor(origin_sig_bits, destination_sig_bits)

        if not any(non_equal_sig_bits):
            return I ^ num_qubits

        # I am deeply sorry for this code, but I don't know another way to do it.
        sig_in_origin_only_indices = np.extract(
            np.logical_and(non_equal_sig_bits, origin_sig_bits),
            np.arange(num_qubits))
        sig_in_dest_only_indices = np.extract(
            np.logical_and(non_equal_sig_bits, destination_sig_bits),
            np.arange(num_qubits))

        if len(sig_in_origin_only_indices) > 0 and len(sig_in_dest_only_indices) > 0:
            origin_anchor_bit = min(sig_in_origin_only_indices)
            dest_anchor_bit = min(sig_in_dest_only_indices)
        else:
            # Set to lowest equal bit
            origin_anchor_bit = min(np.extract(sig_equal_sig_bits, np.arange(num_qubits)))
            dest_anchor_bit = origin_anchor_bit

        cnots = QuantumCircuit(num_qubits)
        # Step 3) Take the indices of bits which are sig_bits in
        # pauli but but not in dest, and cnot them to the pauli anchor.
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

        return PrimitiveOp(cnots.to_instruction())

    # TODO change to only accept PrimitiveOp Pauli.
    def get_cob_circuit(self, origin: Union[Pauli, PauliOp]) -> (PrimitiveOp, PauliOp):
        """ The goal of this module is to construct a circuit which maps the +1 and -1 eigenvectors
        of the origin pauli to the +1 and -1 eigenvectors of the destination pauli. It does so by
            1) converting any |i+⟩ or |i+⟩ eigenvector bits in the origin to
             |+⟩ and |-⟩ with S†s, then
            2) converting any |+⟩ or |+⟩ eigenvector bits in the converted origin to
             |0⟩ and |1⟩ with Hs, then
             3) writing the parity of the significant (Z-measured, rather than
            I) bits in the origin to a single
            "origin anchor bit," using cnots, which will hold the parity of these bits,
            4) swapping the parity of the pauli anchor bit into a destination anchor bit using
            a swap gate (only if they are different, if there are any bits which are significant
            in both origin and dest, we set both anchors to one of these bits to avoid a swap).
            5) flipping the state (parity) of the destination anchor if the parity of the number
            of pauli significant
            bits is different from the parity of the number of destination significant bits
            (to be flipped back in step 7)
            6) writing the parity of the destination anchor bit into the other significant bits
            of the destination,
            7) flipping back the parity of the destination anchor if we flipped it in step 5)
            8) converting the |0⟩ and |1⟩ significant eigenvector bits to |+⟩ and |-⟩ eigenvector
            bits in the destination where the destination demands it
            (e.g. pauli.x == true for a bit), using Hs
            8) converting the |+⟩ and |-⟩ significant eigenvector bits to
            |i+⟩ and |i-⟩ eigenvector bits in the
            destination where the destination demands it
            (e.g. pauli.x == true and pauli.z == true for a bit), using Ss
        """

        # If pauli is an PrimitiveOp, extract the Pauli
        if isinstance(origin, Pauli):
            origin = PauliOp(origin)

        if not isinstance(origin, PauliOp):
            raise TypeError(
                'PauliBasisChange can only convert Pauli-based OpPrimitives, not {}'.format(type(
                    PrimitiveOp.primitive)))

        # If no destination specified, assume nearest Pauli in {Z,I}^n basis,
        # the standard basis change for expectations.
        destination = self.destination or self.get_diagonal_pauli_op(origin)

        # Pad origin or destination if either are not as long as the other
        origin, destination = self.pad_paulis_to_equal_length(origin, destination)

        origin_sig_bits = np.logical_or(origin.primitive.x, origin.primitive.z)
        destination_sig_bits = np.logical_or(destination.primitive.x, destination.primitive.z)
        if not any(origin_sig_bits) or not any(destination_sig_bits):
            if not (any(origin_sig_bits) or any(destination_sig_bits)):
                # Both all Identity, just return Identities
                return I ^ origin.num_qubits, destination
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
