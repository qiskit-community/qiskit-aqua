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

"""TODO"""

from abc import abstractmethod


class MinEigenSolver:
    """Base class for algorithms that search the minimal eigenvalue of an operator."""

    @abstractmethod
    def __init__(self, operator=None):
        self._operator = operator

    @abstractmethod
    def compute_min_eigenvalue(self, operator=None):
        raise NotImplementedError()

    # Cannot implement this, since ExactEigenSolver and VQE both inherit from
    # QuantumAlgorithm which has the signature `run(self, quantum_instance, kwargs)`
    # and not `run(self, operator)`.
    # @abstractmethod
    # def run(self, operator):
    #     raise NotImplementedError()
