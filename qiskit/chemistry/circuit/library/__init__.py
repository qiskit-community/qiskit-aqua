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
===================================================================
Chemistry Circuit Library (:mod:`qiskit.chemistry.circuit.library`)
===================================================================

A collection of circuits used as building blocks or inputs of algorithms in chemistry.

.. currentmodule:: qiskit.chemistry.circuit.library

Initial states
==============

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   HartreeFock
   VSCF

"""

from .initial_states import (
    HartreeFock,
    VSCF
)

__all__ = ['HartreeFock', 'VSCF']
