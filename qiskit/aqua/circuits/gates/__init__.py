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

"""
Gates (:mod:`qiskit.aqua.circuits.gates`)
=========================================
A collection of useful gates that may be used to build quantum algorithms
and components. These gates are created and *monkey patched* to Terra
QuantumCircuit class such that can be used similarly to the gates that are
supplied by Terra e.g. `qc.mcu1(theta, controls_list, target)`.

.. currentmodule:: qiskit.aqua.circuits.gates

Gates
=====

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

    mcmt

"""

from .multi_control_multi_target_gate import mcmt

__all__ = [
    'mcmt',
]
