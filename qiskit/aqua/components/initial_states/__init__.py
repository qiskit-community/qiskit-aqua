# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2020.
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
Initial states are a fixed quantum state. These can be used, for example, to define a starting
state for :mod:`~qiskit.aqua.components.variational_forms`, such as when used with
:class:`~qiskit.aqua.algorithms.VQE`, or to define a starting state for the evolution
in algorithms such as :class:`~qiskit.aqua.algorithms.QPE` and
:class:`~qiskit.aqua.algorithms.IQPE`.

If you have a specific quantum circuit you would like to use as an initial state, and do not
want to make a new class derived from :class:`InitialState` to use it, this can be especially so
if the circuit is fixed, then see :class:`Custom` which allows it to be used as an
:class:`InitialState` for algorithms and components that expect this as a type.

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
