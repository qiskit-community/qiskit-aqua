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

"""The CVaR (Conditional Value at Risk) expectation class."""

import logging
from typing import Union, Optional

from ..operator_base import OperatorBase
from ..list_ops import ListOp, ComposedOp
from ..state_fns import CVaRMeasurement, OperatorStateFn
from .expectation_base import ExpectationBase
from .pauli_expectation import PauliExpectation
from .aer_pauli_expectation import AerPauliExpectation

logger = logging.getLogger(__name__)


class CVaRExpectation(ExpectationBase):
    r"""Compute the Conditional Value at Risk (CVaR) expectation value.

    The standard approach to calculating the expectation value of a Hamiltonian w.r.t. a
    state is to take the sample mean of the measurement outcomes. Instead, for a diagonal
    Hamiltonian, we use CVaR as an aggregation function instead of the mean, as proposed in [1].
    It is empirically shown, that this can lead to faster convergence for combinatorial
    optimization problems.

    Examples:

        >>> from qiskit import Aer
        >>> from qiskit.aqua.operators import Z, I, Plus, StateFn, CVarExpectation, CircuitSampler
        >>> operator = Z ^ I ^ Z ^ I
        >>> state = Plus ^ 4
        >>> op = ~StateFn(operator) @ state
        >>> cvar_expecation = CVaRExpectation(alpha=0.1).convert(op)
        >>> exact_value = cvar_expecation.eval()
        >>> sampler = CircuitSampler(Aer.get_backend('qasm_simulator'))
        >>> sampled_value = sampler.convert(cvar_expectation).eval()

    References:

        [1]: Barkoutsos, P. K., Nannicini, G., Robert, A., Tavernelli, I., and Woerner, S.,
             "Improving Variational Quantum Optimization using CVaR"
             `arXiv:1907.04769 <https://arxiv.org/abs/1907.04769>`_

    """

    def __init__(self, alpha: float, expectation: Optional[ExpectationBase] = None) -> None:
        """
        Args:
            alpha: The alpha value describing the quantile considered in the expectation value.
            expectation: An expectation object to compute the expectation value. Defaults
                to the PauliExpectation calculation.

        Raises:
            NotImplementedError: If the ``expectation`` is an AerPauliExpecation.
        """
        self.alpha = alpha
        if isinstance(expectation, AerPauliExpectation):
            raise NotImplementedError('AerPauliExpecation currently not supported.')
        if expectation is None:
            expectation = PauliExpectation()
        self.expectation = expectation

    def convert(self, operator: OperatorBase) -> OperatorBase:
        """Return an expression that computes the CVaR expectation upon calling ``eval``.

        Args:
            operator: The operator to convert.

        Returns:
            The converted operator.
        """
        expectation = self.expectation.convert(operator)

        # replace OperatorMeasurements by CVaRMeasurement
        def replace_with_cvar(operator):
            if isinstance(operator, OperatorStateFn) and operator.is_measurement:
                return CVaRMeasurement(operator.primitive, alpha=self.alpha)
            elif isinstance(operator, ListOp):
                return operator.traverse(replace_with_cvar)
            return operator

        return replace_with_cvar(expectation)

    def compute_variance(self, exp_op: OperatorBase) -> Union[list, float]:
        r"""
        Compute the variance of the expectation estimator. Because this expectation
        works by matrix multiplication, the estimation is exact and the variance is
        always 0, but we need to return those values in a way which matches the Operator's
        structure.

        Args:
            exp_op: The full expectation value Operator.

        Returns:
             The variances or lists thereof (if exp_op contains ListOps) of the expectation value
             estimation, equal to 0.
        """

        # Need to do this to mimic Op structure
        def sum_variance(operator):
            if isinstance(operator, ComposedOp):
                return 0.0
            elif isinstance(operator, ListOp):
                return operator._combo_fn([sum_variance(op) for op in operator.oplist])
            else:
                return 0.0

        return sum_variance(exp_op)
