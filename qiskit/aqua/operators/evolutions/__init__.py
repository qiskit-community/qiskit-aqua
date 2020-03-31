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

"""
Operator Evolution algorithms - Algorithms for producing or approximating
the exponential of an operator.
"""

from .evolution_base import EvolutionBase
from .evolved_op import EvolvedOp
from .pauli_trotter_evolution import PauliTrotterEvolution
from .matrix_evolution import MatrixEvolution
from .trotterizations import TrotterizationBase, Trotter, Suzuki, QDrift

# TODO matrix evolution
# TODO quantum signal processing
# TODO evolve by density matrix (need to add iexp to operator_state_fn)
# TODO linear combination evolution

__all__ = ['EvolutionBase',
           'EvolvedOp',
           'PauliTrotterEvolution',
           'MatrixEvolution',
           'TrotterizationBase',
           'Trotter',
           'Suzuki',
           'QDrift']
