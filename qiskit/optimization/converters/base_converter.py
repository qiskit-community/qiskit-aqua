# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""An abstract class for optimization algorithms in Qiskit's optimization module."""

from abc import ABC, abstractmethod

from ..algorithms.optimization_algorithm import OptimizationResult
from ..problems.quadratic_program import QuadraticProgram


class BaseConverter(ABC):
    """
    An abstract class for converters of quadratic programs in Qiskit's optimization module.
    """

    @abstractmethod
    def convert(self, problem: QuadraticProgram) -> QuadraticProgram:
        """ convert an QuadratciProgram into an QuadraticProgram with a specific form"""
        raise NotImplementedError

    @abstractmethod
    def interpret(self, result: OptimizationResult) -> OptimizationResult:
        """ interpret an OptimizationResult based on the original QuadraticProgram"""
        raise NotImplementedError
