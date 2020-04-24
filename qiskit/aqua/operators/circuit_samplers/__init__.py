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

"""
Circuit Samplers (:mod:`qiskit.aqua.operators.circuit_samplers`)
================================================================
Converters for replacing
:class:`~qiskit.aqua.operators.state_functions.CircuitStateFn` objects with
:class:`~qiskit.aqua.operators.state_functions.DictStateFn` objects representing samples
of the :class:`~qiskit.aqua.operators.state_functions.StateFn`.

.. currentmodule:: qiskit.aqua.operators.circuit_samplers

Circuit Sampler Base Class
==========================

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   CircuitSamplerBase

Circuit Samplers
================

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   CircuitSamplerFactory
   CircuitSampler
   IBMQSampler

"""

from .circuit_sampler_base import CircuitSamplerBase
from .circuit_sampler_factory import CircuitSamplerFactory
from .ibmq_sampler import IBMQSampler

__all__ = ['CircuitSamplerBase',
           'CircuitSamplerFactory',
           'IBMQSampler']
