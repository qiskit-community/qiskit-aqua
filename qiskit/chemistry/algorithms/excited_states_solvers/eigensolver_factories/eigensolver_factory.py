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

"""The eigensolver factory for excited states calculation algorithms."""

from abc import ABC, abstractmethod
from qiskit.aqua.algorithms import Eigensolver
from qiskit.chemistry.transformations import Transformation


class EigensolverFactory(ABC):
    """A factory to construct a eigensolver based on a qubit operator transformation."""

    @abstractmethod
    def get_solver(self, transformation: Transformation) -> Eigensolver:
        """Returns a eigensolver, based on the qubit operator transformation.

        Args:
            transformation: The qubit operator transformation.

        Returns:
            An eigensolver suitable to compute the excited states of the molecule transformed
            by ``transformation``.
        """
        raise NotImplementedError
