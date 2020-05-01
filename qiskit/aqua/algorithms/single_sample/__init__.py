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

""" single sample packages """

from .grover.grover import Grover
from .iterative_qpe.iqpe import IQPE
from .qpe.qpe import QPE
from .amplitude_estimation.ae import AmplitudeEstimation
from .amplitude_estimation.iqae import IterativeAmplitudeEstimation
from .amplitude_estimation.mlae import MaximumLikelihoodAmplitudeEstimation
from .simon.simon import Simon
from .deutsch_jozsa.dj import DeutschJozsa
from .bernstein_vazirani.bv import BernsteinVazirani
from .hhl.hhl import HHL
from .shor.shor import Shor


__all__ = [
    'Grover',
    'IQPE',
    'QPE',
    'AmplitudeEstimation',
    'IterativeAmplitudeEstimation',
    'MaximumLikelihoodAmplitudeEstimation',
    'Simon',
    'DeutschJozsa',
    'BernsteinVazirani',
    'HHL',
    'Shor',
]
