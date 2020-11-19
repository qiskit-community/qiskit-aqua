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
Uncertainty Models (:mod:`qiskit.aqua.components.uncertainty_models`)
=====================================================================
An uncertainty model is an implementation that provides a way to construct a
quantum circuit to prepare a state which corresponds to a particular distribution.
More precisely, the resulting state together with an affine map can be used to sample from the
considered distribution. The qubits are measured and then mapped to the desired range using
the affine map.

.. currentmodule:: qiskit.aqua.components.uncertainty_models

Uncertainty Model Base Classes
==============================
:class:`UncertaintyModel` is the base class from which further
base classes for univariate and multivariate distributions are
derived

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   UncertaintyModel
   UnivariateDistribution
   MultivariateDistribution

Univariate Distributions
========================

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

    NormalDistribution
    LogNormalDistribution
    BernoulliDistribution
    UniformDistribution
    UnivariateVariationalDistribution

Multivariate Distributions
==========================

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

    MultivariateNormalDistribution
    MultivariateLogNormalDistribution
    MultivariateUniformDistribution
    MultivariateVariationalDistribution
    GaussianConditionalIndependenceModel

"""

from .uncertainty_model import UncertaintyModel
from .univariate_distribution import UnivariateDistribution
from .multivariate_distribution import MultivariateDistribution
from .normal_distribution import NormalDistribution
from .log_normal_distribution import LogNormalDistribution
from .bernoulli_distribution import BernoulliDistribution
from .uniform_distribution import UniformDistribution
from .multivariate_normal_distribution import MultivariateNormalDistribution
from .multivariate_log_normal_distribution import MultivariateLogNormalDistribution
from .multivariate_uniform_distribution import MultivariateUniformDistribution
from .univariate_variational_distribution import UnivariateVariationalDistribution
from .multivariate_variational_distribution import MultivariateVariationalDistribution
from .gaussian_conditional_independence_model import GaussianConditionalIndependenceModel

__all__ = ['UncertaintyModel',
           'UnivariateDistribution',
           'MultivariateDistribution',
           'NormalDistribution',
           'LogNormalDistribution',
           'BernoulliDistribution',
           'UniformDistribution',
           'MultivariateNormalDistribution',
           'MultivariateLogNormalDistribution',
           'MultivariateUniformDistribution',
           'UnivariateVariationalDistribution',
           'MultivariateVariationalDistribution',
           'GaussianConditionalIndependenceModel']
