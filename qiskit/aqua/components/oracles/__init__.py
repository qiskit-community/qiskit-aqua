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
Oracles (:mod:`qiskit.aqua.components.oracles`)
===================================================================
Oracles are used by :class:`Grover` and :class:`DeutschJosza` algorithms
for example.

.. currentmodule:: qiskit.aqua.components.oracles

Oracle Base Class
=================

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   Oracle

Oracles
=======

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   TruthTableOracle
   LogicalExpressionOracle
   CustomCircuitOracle

"""

from .oracle import Oracle
from .truth_table_oracle import TruthTableOracle
from .logical_expression_oracle import LogicalExpressionOracle
from .custom_circuit_oracle import CustomCircuitOracle


__all__ = [
    'Oracle',
    'TruthTableOracle',
    'LogicalExpressionOracle',
    'CustomCircuitOracle',
]
