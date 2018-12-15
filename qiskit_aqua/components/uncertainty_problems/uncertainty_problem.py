# -*- coding: utf-8 -*-

# Copyright 2018 IBM.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""
The abstract Uncertainty Problem pluggable component.
"""

from abc import ABC
from qiskit_aqua import Pluggable
from qiskit_aqua.utils import CircuitFactory


class UncertaintyProblem(CircuitFactory, Pluggable, ABC):
    """
    The abstract Uncertainty Problem pluggable component.
    """

    @classmethod
    def init_params(cls, params):
        args = {k: v for k, v in params.items() if k != 'name'}
        return cls(**args)

    def __init__(self, num_qubits):
        super().__init__(num_qubits)

    def value_to_estimation(self, value):
        return value
