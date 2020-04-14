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

""" Factory for evolution algorithms """

import logging

from ..operator_base import OperatorBase
from .evolution_base import EvolutionBase
from .pauli_trotter_evolution import PauliTrotterEvolution
from .matrix_evolution import MatrixEvolution

logger = logging.getLogger(__name__)


class EvolutionFactory():
    """ A factory for convenient construction of Evolution algorithms.
    """

    @staticmethod
    # pylint: disable=inconsistent-return-statements
    def build(operator: OperatorBase = None) -> EvolutionBase:
        """
        Args:
            operator: the operator being evolved
        Returns:
            EvolutionBase: derived class
        Raises:
            ValueError: evolutions of Mixed Operators not yet supported.
        """
        # pylint: disable=cyclic-import,import-outside-toplevel
        primitives = operator.get_primitives()
        if 'Pauli' in primitives:
            # TODO figure out what to do based on qubits and hamming weight.
            return PauliTrotterEvolution()

        # TODO
        elif 'Matrix' in primitives:
            return MatrixEvolution()

        else:
            raise ValueError('Evolutions of Mixed Operators not yet supported.')
