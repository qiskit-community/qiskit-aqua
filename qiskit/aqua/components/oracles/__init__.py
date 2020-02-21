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

r"""
Oracles (:mod:`qiskit.aqua.components.oracles`)
===================================================================
An oracle is a black box operation used as input to another algorithm.
They tend to encode a function :math:`f:\{0,1\}^n \rightarrow \{0,1\}^m`
where the goal of the algorithm is to determine some property of :math:`f`.

Oracles are used by :class:`~qiskit.aqua.algorithms.Grover` and
:class:`~qiskit.aqua.algorithms.DeutschJozsa` algorithms for example.

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

   LogicalExpressionOracle
   TruthTableOracle
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
