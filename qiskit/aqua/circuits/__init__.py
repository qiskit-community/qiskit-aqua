# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Circuits (:mod:`qiskit.aqua.circuits`)
======================================
Collection of circuits and gates that may be used to build quantum algorithms
and components.

Note:
    As of Aqua 0.7.0 Gates that were formerly here such as `mct` etc., that were initially built
    out to facilitate the development of Aqua algorithms, have been moved into Terra.

    Likewise there are Circuits here, that are now deprecated, which have been moved and have
    updated versions in Terra `qiskit.circuit.library` which should be used for any future work.
    The circuit documentation here indicates the corresponding replacement circuit in the library.

.. currentmodule:: qiskit.aqua.circuits

Circuits
========

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

    CNF
    DNF
    ESOP
    PhaseEstimationCircuit
    StateVectorCircuit
    FourierTransformCircuits
    FixedValueComparator
    LinearRotation
    PiecewiseLinearRotation
    PolynomialRotation
    WeightedSumOperator

"""

from .boolean_logical_circuits import CNF, DNF, ESOP
from .phase_estimation_circuit import PhaseEstimationCircuit
from .statevector_circuit import StateVectorCircuit
from .fourier_transform_circuits import FourierTransformCircuits
from .fixed_value_comparator import FixedValueComparator
from .linear_rotation import LinearRotation
from .piecewise_linear_rotation import PiecewiseLinearRotation
from .polynomial_rotation import PolynomialRotation
from .weighted_sum_operator import WeightedSumOperator

__all__ = [
    'CNF',
    'DNF',
    'ESOP',
    'PhaseEstimationCircuit',
    'StateVectorCircuit',
    'FourierTransformCircuits',
    'FixedValueComparator',
    'LinearRotation',
    'PiecewiseLinearRotation',
    'PolynomialRotation',
    'WeightedSumOperator'
]
