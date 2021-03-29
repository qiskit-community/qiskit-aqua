# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
=========================================================
Qiskit's optimization module (:mod:`qiskit.optimization`)
=========================================================

.. currentmodule:: qiskit.optimization

Qiskit's optimization module covers the whole range from high-level modeling of optimization
problems, with automatic conversion of problems to different required representations,
to a suite of easy-to-use quantum optimization algorithms that are ready to run on
classical simulators, as well as on real quantum devices via Qiskit.

This module enables easy, efficient modeling of optimization problems using `docplex
<https://developer.ibm.com/docloud/documentation/optimization-modeling/modeling-for-python/>`_.
A uniform interface as well as automatic conversion between different problem representations
allows users to solve problems using a large set of algorithms, from variational quantum algorithms,
such as the Quantum Approximate Optimization Algorithm
(:class:`~qiskit.aqua.algorithms.QAOA`), to
`Grover Adaptive Search <https://arxiv.org/abs/quant-ph/9607014>`_
(:class:`~algorithms.GroverOptimizer`), leveraging
fundamental :mod:`~qiskit.aqua.algorithms` provided by Qiskit Aqua. Furthermore, the modular design
of the optimization module allows it to be easily extended and facilitates rapid development and
testing of new algorithms. Compatible classical optimizers are also provided for testing,
validation, and benchmarking.

Qiskit's optimization module supports Quadratically Constrained Quadratic Programs – for simplicity
we refer to them just as Quadratic Programs – with binary, integer, and continuous variables, as
well as equality and inequality constraints. This class of optimization problems has a vast amount
of relevant applications, while still being efficiently representable by matrices and vectors.
This class covers some very interesting sub-classes, from Convex Continuous Quadratic Programs,
which can be solved efficiently by classical optimization algorithms, to Quadratic Unconstrained
Binary Optimization QUBO) problems, which cover many NP-complete, i.e., classically intractable,
problems.

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

    QuadraticProgram

Representation of a Quadratically Constrained Quadratic Program supporting inequality and
equality constraints as well as continuous, binary, and integer variables.

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

    QiskitOptimizationError

In addition to standard Python errors the optimization module will raise this error if circumstances
are that it cannot proceed to completion.

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

    INFINITY

A constant for infinity.

Submodules
==========

.. autosummary::
   :toctree:

   algorithms
   applications
   converters
   problems

"""

from qiskit.aqua.deprecation import warn_package
from .infinity import INFINITY  # must be at the top of the file
from .exceptions import QiskitOptimizationError
from .problems.quadratic_program import QuadraticProgram
from ._logging import (get_qiskit_optimization_logging,
                       set_qiskit_optimization_logging)

warn_package('optimization', 'qiskit_optimization', 'qiskit-optimization')

__all__ = ['QuadraticProgram',
           'QiskitOptimizationError',
           'get_qiskit_optimization_logging',
           'set_qiskit_optimization_logging',
           'INFINITY'
           ]
