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


Feature Maps
============
In machine learning, pattern recognition and image processing, a **feature map**
starts from an initial set of measured data and builds derived values (also known as
**features**) intended to be informative and non-redundant, facilitating the subsequent
learning and generalization steps, and in some cases leading to better human
interpretations.
A feature map is related to **dimensionality reduction**; it involves reducing the amount of
resources required to describe a large set of data. When performing analysis of complex data,
one of the major problems stems from the number of variables involved. Analysis with a large
number of variables generally requires a large amount of memory and computation power, and may
even cause a classification algorithm to overfit to training samples and generalize poorly to new
samples.
When the input data to an algorithm is too large to be processed and is suspected to be redundant
(for example, the same measurement is provided in both pounds and kilograms), then it can be
transformed into a reduced set of features, named a **feature vector**.
The process of determining a subset of the initial features is called **feature selection**.
The selected features are expected to contain the relevant information from the input data,
so that the desired task can be performed by using the reduced representation instead
of the complete initial data.

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   PauliExpansion
   PauliZExpansion
   FirstOrderExpansion
   SecondOrderExpansion
   RawFeatureVector

Feature Map Utility
===================

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   self_product

"""

from .ansatz import Ansatz
from .operator_ansatz import OperatorAnsatz
from .two_local_ansatz import TwoLocalAnsatz
from .ry import RY
from .ryrz import RYRZ
from .swaprz import SwapRZ
from .feature_maps import (PauliExpansion, PauliZExpansion, FirstOrderExpansion,
                           SecondOrderExpansion, RawFeatureVector, self_product)

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
    'self_product',
]
