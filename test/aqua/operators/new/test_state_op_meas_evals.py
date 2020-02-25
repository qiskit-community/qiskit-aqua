# -*- coding: utf-8 -*-

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

""" Test Operator construction, including OpPrimitives and singletons. """

import unittest
import itertools
import numpy as np

from qiskit import QuantumCircuit, BasicAer, execute, ClassicalRegister
from qiskit.quantum_info import Statevector

from test.aqua import QiskitAquaTestCase
from qiskit.aqua.operators import StateFn, Zero, One, Plus, Minus, OpPrimitive, H, I, Z, X, Y


class TestStateOpMeasEvals(QiskitAquaTestCase):
    """Tests of evals of Meas-Operator-StateFn combos."""

    def test_statefn_dicts(self):
        wf1 = Zero