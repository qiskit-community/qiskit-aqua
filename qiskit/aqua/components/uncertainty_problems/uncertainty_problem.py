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
from qiskit.aqua import AquaError, Pluggable, PluggableType, get_pluggable_class
from qiskit.aqua.utils import CircuitFactory


class UncertaintyProblem(CircuitFactory, Pluggable, ABC):
    """
    The abstract Uncertainty Problem pluggable component.
    """

    @classmethod
    def init_params(cls, params):
        uncertainty_problem_params = params.get(Pluggable.SECTION_KEY_UNCERTAINTY_PROBLEM)
        args = {k: v for k, v in uncertainty_problem_params.items() if k != 'name'}

        # Uncertainty problems take an uncertainty model. Each can take a specific type as
        # a dependent. We currently have two known types and to save having init_params in
        # each of the problems a problem can use this base class method that tries to find
        # params for the set of known uncertainty model types.
        uncertainty_model_params = params.get(Pluggable.SECTION_KEY_UNIVARIATE_DISTRIBUTION)
        pluggable_type = PluggableType.UNIVARIATE_DISTRIBUTION
        if uncertainty_model_params is None:
            uncertainty_model_params = params.get(Pluggable.SECTION_KEY_MULTIVARIATE_DISTRIBUTION)
            pluggable_type = PluggableType.MULTIVARIATE_DISTRIBUTION
        if uncertainty_model_params is None:
            raise AquaError("No params for known uncertainty model types found")
        uncertainty_model = get_pluggable_class(pluggable_type,
                                                uncertainty_model_params['name']).init_params(params)

        return cls(uncertainty_model, **args)

    def __init__(self, num_qubits):
        super().__init__(num_qubits)

    def value_to_estimation(self, value):
        return value
