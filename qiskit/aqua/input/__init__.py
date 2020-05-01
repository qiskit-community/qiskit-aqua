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

""" Input Packages """

from .algorithm_input import AlgorithmInput
from .energy_input import EnergyInput
from .linear_system_input import LinearSystemInput
from .classification_input import ClassificationInput
from .qgan_input import QGANInput

__all__ = ['AlgorithmInput',
           'EnergyInput',
           'LinearSystemInput',
           'ClassificationInput',
           'QGANInput']
