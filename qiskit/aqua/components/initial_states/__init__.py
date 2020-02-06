# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Initial States (:mod:`qiskit.aqua.components.initial_states`)
=============================================================
Initial states are a state ansatz that can be used to define a
starting state for example with VQE, in conjunction with a
variational form, or with (I)QPE algorithms

.. currentmodule:: qiskit.aqua.components.initial_states

Initial State Base Class
========================

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   InitialState

Initial States
==============

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   Custom
   VarFormBased
   Zero

"""

from .initial_state import InitialState
from .custom import Custom
from .var_form_based import VarFormBased
from .zero import Zero

__all__ = ['InitialState',
           'Custom',
           'VarFormBased',
           'Zero'
           ]
