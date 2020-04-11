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

"""Quantum Phase Estimation Circuit."""

import warnings
from qiskit.aqua.algorithms import PhaseEstimationCircuit

warnings.warn('The qiskit.aqua.circuits.phase_estimation_circuit module is deprecated as of 0.7.0 '
              'and will be removed no earlier than 3 months after the release. '
              'The PhaseEstimationCircuit was moved to qiskit.aqua.algorithms.',
              DeprecationWarning, stacklevel=2)

__all__ = ['PhaseEstimationCircuit']
