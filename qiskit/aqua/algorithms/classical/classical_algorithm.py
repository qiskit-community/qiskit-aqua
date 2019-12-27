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
This module implements the abstract base class for classical algorithm modules.

To create add-on classical algorithm modules subclass the ClassicalAlgorithm
class in this module.
Doing so requires that the required algorithm interface is implemented.
"""

# pylint: disable=unused-import

from abc import abstractmethod
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import QuantumAlgorithm


class ClassicalAlgorithm(QuantumAlgorithm):
    """
    Base class for Classical Algorithms.

    This method should initialize the module and
    use an exception if a component of the module is available.
    """

    @abstractmethod
    def __init__(self):
        super().__init__()

    def run(self, quantum_instance=None, **kwargs):
        """Execute the algorithm with selected backend.

        Args:
            quantum_instance (QuantumInstance or BaseBackend): the experimental setting.
            kwargs (dict): kwargs
        Returns:
            dict: results of an algorithm.
        """

        return self._run()
