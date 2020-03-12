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
Expectation Value algorithms - Algorithms for approximating the value of some function
over a probability distribution, or in the quantum case, algorithms for approximating
the value of some observable over a state function.

"""

from .expectation_base import ExpectationBase
from .pauli_expectation import PauliExpectation
from .aer_pauli_expectation import AerPauliExpectation
from .matrix_expectation import MatrixExpectation

__all__ = ['ExpectationBase',
           'PauliExpectation',
           'AerPauliExpectation',
           'MatrixExpectation']
