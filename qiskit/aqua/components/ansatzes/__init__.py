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
Ansätze (:mod:`qiskit.aqua.components.ansatzes`)
===================================================================
In quantum mechanics, the *variational method* is one way of finding approximations to the lowest
energy eigenstate, or *ground state*, and some excited states. This allows calculating approximate
wave functions, such as molecular orbitals. The basis for this method is the *variational
principle*.

The variational method consists of choosing a *trial wave function*, or *ansatz* (or
*variational form*), that depends on one or more parameters, and finding the values of these
parameters for which the expectation value of the energy is the lowest possible. The wave function
obtained by fixing the parameters to such values is then an approximation to the ground state wave
function, and the expectation value of the energy in that state is an upper bound to the ground
state energy. Quantum variational algorithms, such as :class:`~qiskit.aqua.algorithms.VQE`,
apply the variational method.

As such, they require an ansatz.

.. currentmodule:: qiskit.aqua.components.ansatzes

Submodules
==========

.. autosummary::
   :toctree:

   feature_maps

Ansatz Base Class
=================

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   Ansatz

Ansätze
=======

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   TwoLocalAnsatz
   RY
   RYRZ
   SwapRZ

"""
from .ansatz import Ansatz
from .operator_ansatz import OperatorAnsatz
from .two_local_ansatz import TwoLocalAnsatz
from .ry import RY
from .ryrz import RYRZ
from .swaprz import SwapRZ
from .feature_maps import (PauliExpansion, FirstOrderExpansion, SecondOrderExpansion,
                           RawFeatureVector)

__all__ = [
    'Ansatz',
    'FirstOrderExpansion',
    'OperatorAnsatz',
    'PauliExpansion',
    'RawFeatureVector',
    'RY',
    'RYRZ',
    'SecondOrderExpansion',
    'SwapRZ',
    'TwoLocalAnsatz',
]
