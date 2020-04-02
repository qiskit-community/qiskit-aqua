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

"""This file implements the OptimizationResult, returned by the optimization algorithms."""

from typing import Any, Optional


class OptimizationResult:
    """The optimization result class.

    The optimization algorithms return an object of the type `OptimizationResult`, which enforces
    providing the following attributes.

    Attributes:
        x: The optimal value found in the optimization algorithm.
        fval: The function value corresponding to the optimal value.
        results: The original results object returned from the optimization algorithm. This can
            contain more information than only the optimal value and function value.
    """

    def __init__(self, x: Optional[Any] = None, fval: Optional[Any] = None,
                 results: Optional[Any] = None) -> None:
        """Initialize the optimization result."""
        self._val = x
        self._fval = fval
        self._results = results

    def __repr__(self):
        return '([%s] / %s)' % (','.join([str(x_) for x_ in self.x]), self.fval)

    @property
    def x(self) -> Any:
        """Returns the optimal value found in the optimization.

        Returns:
            The optimal value found in the optimization.
        """
        return self._val

    @property
    def fval(self) -> Any:
        """Returns the optimal function value.

        Returns:
            The function value corresponding to the optimal value found in the optimization.
        """
        return self._fval

    @property
    def results(self) -> Any:
        """Return the original results object from the algorithm.

        Currently a dump for any leftovers.

        Returns:
            Additional result information of the optimization algorithm.
        """
        return self._results

    @x.setter
    def x(self, x: Any) -> None:
        """Set a new optimal value.

        Args:
            x: The new optimal value.
        """
        self._val = x

    @fval.setter
    def fval(self, fval: Any) -> None:
        """Set a new optimal function value.

        Args:
            fval: The new optimal function value.
        """
        self._fval = fval

    @results.setter
    def results(self, results: Any) -> None:
        """Set results.

        Args:
            results: The new additional results of the optimization.
        """
        self._results = results
