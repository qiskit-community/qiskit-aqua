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

import logging

from ..operator_base import OperatorBase
from ..converters import ConverterBase

logger = logging.getLogger(__name__)


class EvolutionBase(ConverterBase):
    """ A base for Evolution algorithms. An evolution algorithm is a converter which recurses
    through an operator tree,
    replacing the EvolvedOps with a backend-runnable Hamiltonian simulation equaling
    or approximating the
    exponentiation of its contained operator.

    """

    @staticmethod
    # pylint: disable=inconsistent-return-statements
    def factory(operator: OperatorBase = None) -> ConverterBase:
        """
        Args:
        Returns:
            EvolutionBase: derived class
        Raises:
            ValueError: evolutions of Mixed Operators not yet supported.
        """
        # pylint: disable=cyclic-import,import-outside-toplevel
        # TODO remove state from factory and inits?
        primitives = operator.get_primitives()
        if 'Pauli' in primitives:
            # TODO figure out what to do based on qubits and hamming weight.
            from .pauli_trotter_evolution import PauliTrotterEvolution
            return PauliTrotterEvolution()

        # TODO
        elif 'Matrix' in primitives:
            from .matrix_evolution import MatrixEvolution
            return MatrixEvolution()

        # TODO
        # elif primitives == {'QuantumCircuit'}:
        #     from .density_matrix_evolution import DensityMatrixEvolution
        #     return DensityMatrixEvolution()

        else:
            raise ValueError('Evolutions of Mixed Operators not yet supported.')

    def convert(self, operator: OperatorBase) -> OperatorBase:
        raise NotImplementedError

    # TODO @abstractmethod
    # def error_bounds(self):
    #     """ error bounds """
    #     raise NotImplementedError
