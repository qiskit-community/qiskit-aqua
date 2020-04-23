# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Interface for all objects that have a parent QuadraticProgram."""


class HasQuadraticProgram:
    """Abstract interface class for all objects that have a parent QuadraticProgram."""

    def __init__(self, quadratic_program: "QuadraticProgram") -> None:
        """ Initialize object with parent QuadraticProgram.

        Args:
            quadratic_program: The parent QuadraticProgram.
        """
        self._quadratic_program = quadratic_program

    @property
    def quadratic_program(self) -> "QuadraticProgram":
        """Returns the parent QuadraticProgram.

        Returns:
            The parent QuadraticProgram.
        """
        return self._quadratic_program

    @quadratic_program.setter
    def quadratic_program(self, quadratic_program: "QuadraticProgram") -> None:
        """Sets the parent QuadraticProgram.

        Args:
            quadratic_program: The parent QuadraticProgram.
        """
        self._quadratic_program = quadratic_program
