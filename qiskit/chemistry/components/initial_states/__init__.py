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
Initial States (:mod:`qiskit.chemistry.components.initial_states`)
==================================================================
These are chemistry specific Aqua Initial States where they inherit from
Aqua :class:`~qiskit.aqua.components.initial_states.InitialState`.
As they rely on chemistry specific knowledge and/or functions they live here rather than in Aqua.

.. currentmodule:: qiskit.chemistry.components.initial_states

Initial States
==============

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   HartreeFock

"""

from .hartree_fock import HartreeFock

__all__ = ['HartreeFock']
