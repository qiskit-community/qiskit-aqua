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
This module contains the definition of a base class for
uncertainty models. An uncertainty model could be used for
constructing Amplification Estimation tasks.
"""

from abc import ABC, abstractmethod
from qiskit.aqua import Pluggable
from qiskit.aqua.utils import CircuitFactory


class UncertaintyModel(CircuitFactory, Pluggable, ABC):
    """
    The abstract Uncertainty Model
    """

    @classmethod
    def init_params(cls, params):
        """ init params """
        uncertainty_model_params = params.get(cls.get_section_key_name())
        args = {k: v for k, v in uncertainty_model_params.items() if k != 'name'}
        return cls(**args)

    @classmethod
    @abstractmethod
    def get_section_key_name(cls):
        """ get section key name """
        pass

    # pylint: disable=useless-super-delegation
    def __init__(self, num_target_qubits):
        super().__init__(num_target_qubits)
