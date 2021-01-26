# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Uncertainty Problems (:mod:`qiskit.aqua.components.uncertainty_problems`)
=========================================================================
Uncertainty is present in most realistic applications, and often it is necessary to evaluate
the behavior of a system under uncertain data.
For instance, in finance, it is of interest to evaluate expected value or risk metrics of
financial products that depend on underlying stock prices, economic factors, or changing
interest rates. Classically, such problems are often evaluated using Monte Carlo simulation.
However, Monte Carlo simulation does not converge very fast, which implies that large numbers of
samples are required to achieve estimators of reasonable accuracy and confidence.
In quantum computing, *amplitude estimation* can be used instead, which can lead to a quadratic
speed-up. Thus, millions of classical samples could be replaced by a few thousand quantum samples.

*Amplitude estimation* is a derivative of *quantum phase estimation* applied to a particular
operator :math:`A`. :math:`A` is assumed to operate on (n+1) qubits (+ possible ancillas) where
the n qubits represent the uncertainty (see :mod:`~qiskit.aqua.components.uncertainty_models`) and
the last qubit is used to represent the (normalized) objective value as its amplitude.
In other words, :math:`A` is constructed such that the probability of measuring a '1' in the
objective qubit is equal to the value of interest.
Aqua has several amplitude estimation algorithms:
:class:`~qiskit.aqua.algorithms.AmplitudeEstimation`,
:class:`~qiskit.aqua.algorithms.IterativeAmplitudeEstimation` and
:class:`~qiskit.aqua.algorithms.MaximumLikelihoodAmplitudeEstimation`.

Since the value of interest has to be normalized to lie in [0, 1], an uncertainty problem also
provides a function:

.. code:: python

    def value_to_estimator(self, value):
        return value

which is used to map the result of *amplitude estimation* to the range of interest.
The standard implementation is just the identity and can be overridden when needed.

.. currentmodule:: qiskit.aqua.components.uncertainty_problems

Uncertainty Problem Base Classes
================================
:class:`UncertaintyProblem` is the base class from which further
base classes for univariate and multivariate problems are
derived

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   UncertaintyProblem
   UnivariateProblem
   MultivariateProblem

Univariate Problems
===================

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

    UnivariatePiecewiseLinearObjective

"""

from .uncertainty_problem import UncertaintyProblem
from .multivariate_problem import MultivariateProblem
from .univariate_problem import UnivariateProblem
from .univariate_piecewise_linear_objective import UnivariatePiecewiseLinearObjective

__all__ = ['UncertaintyProblem',
           'MultivariateProblem',
           'UnivariateProblem',
           'UnivariatePiecewiseLinearObjective']
