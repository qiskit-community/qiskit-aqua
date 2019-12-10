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
The abstract Uncertainty Problem pluggable component.
"""

from abc import ABC
from qiskit.aqua import AquaError, Pluggable, PluggableType, get_pluggable_class
from qiskit.aqua.utils import CircuitFactory

# pylint: disable=abstract-method


class UncertaintyProblem(CircuitFactory, Pluggable, ABC):
    """
    The abstract Uncertainty Problem pluggable component.
    """

    @classmethod
    def init_params(cls, params):
        """ init params """
        uncertainty_problem_params = params.get(Pluggable.SECTION_KEY_UNCERTAINTY_PROBLEM)
        args = {k: v for k, v in uncertainty_problem_params.items() if k != 'name'}

        # Uncertainty problems take an uncertainty model. Each can take a specific type as
        # a dependent. We currently have two known types and to save having init_params in
        # each of the problems a problem can use this base class method that tries to find
        # params for the set of known uncertainty model types.
        uncertainty_model_params = params.get(Pluggable.SECTION_KEY_UNIVARIATE_DIST)
        pluggable_type = PluggableType.UNIVARIATE_DISTRIBUTION
        if uncertainty_model_params is None:
            uncertainty_model_params = params.get(Pluggable.SECTION_KEY_MULTIVARIATE_DIST)
            pluggable_type = PluggableType.MULTIVARIATE_DISTRIBUTION
        if uncertainty_model_params is None:
            raise AquaError("No params for known uncertainty model types found")
        uncertainty_model = \
            get_pluggable_class(pluggable_type,
                                uncertainty_model_params['name']).init_params(params)

        return cls(uncertainty_model, **args)

    # pylint: disable=useless-super-delegation
    def __init__(self, num_qubits):
        super().__init__(num_qubits)

    def value_to_estimation(self, value):
        """ value to estimate """
        return value
