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

from .evolution_base import EvolutionBase

from qiskit.aqua.operators import (OpVec, OpSum, OpPauli, OpPrimitive, Z, I, PauliChangeOfBasis)

from . import OpEvolution
from .trotterizations import TrotterizationBase

logger = logging.getLogger(__name__)


class PauliTrotterEvolution(EvolutionBase):
    """ TODO

    """

    def __init__(self, trotter_mode=('suzuki', 2)):
        """
        Args:

        """

        if isinstance(trotter_mode, TrotterizationBase):
            self._trotter = trotter_mode
        else:
            (mode_str, reps) = trotter_mode
            self._trotter = TrotterizationBase.factory(mode=mode_str, reps=reps)

    @property
    def trotter(self):
        return self._trotter

    @trotter.setter
    def trotter(self, trotter):
        self._trotter = trotter

    def evolution_for_pauli(self, pauli_op):
        # TODO Evolve for group of commuting paulis, TODO pauli grouper

        def replacement_fn(cob_instr_op, dest_pauli_op):
            z_evolution = dest_pauli_op.exp_i()
            # Remember, circuit composition order is mirrored operator composition order.
            return cob_instr_op.adjoint().compose(z_evolution).compose(cob_instr_op)

        # Note: PauliChangeOfBasis will pad destination with identities to produce correct CoB circuit
        destination = Z
        cob = PauliChangeOfBasis(destination_basis=destination, replacement_fn=replacement_fn)
        return cob.convert(pauli_op)

    def convert(self, operator):
        if isinstance(operator, OpEvolution):
            if isinstance(operator.primitive, OpSum):
                trotterized = self.trotter.trotterize(operator.primitive)
                return self.convert(trotterized)
            elif isinstance(operator.primitive, OpPauli):
                return self.evolution_for_pauli(operator.primitive)
            # Covers OpVec, OpComposition, OpKron
            elif isinstance(operator.primitive, OpVec):
                converted_ops = [self.convert(op) for op in operator.primitive.oplist]
                return operator.__class__(converted_ops, coeff=operator.coeff)
        elif isinstance(operator, OpVec):
            return operator.traverse(self.convert).reduce()
        else:
            return operator
