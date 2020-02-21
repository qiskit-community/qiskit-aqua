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

""" Test PauliExpectation """

from test.aqua import QiskitAquaTestCase

import numpy as np
import itertools

from qiskit.aqua.operators import X, Y, Z, I, CX, T, H, S, OpPrimitive, OpSum, OpComposition
from qiskit.aqua.operators import StateFn, Zero, Plus, Minus

from qiskit.aqua.algorithms.expectation import ExpectationBase, PauliExpectation
from qiskit import QuantumCircuit, BasicAer


class TestPauliExpectation(QiskitAquaTestCase):
    """Pauli Change of Basis Expectation tests."""

    def test_pauli_expect_single(self):
        op = (Z ^ Z)
        # op = (Z ^ Z)*.5 + (I ^ Z)*.5 + (Z ^ X)*.5
        backend = BasicAer.get_backend('qasm_simulator')
        expect = PauliExpectation(operator=op, backend=backend)
        # wf = (Pl^Pl) + (Ze^Ze)
        wf = CX @ (H^I) @ Zero
        mean = expect.compute_expectation(wf)
        self.assertAlmostEqual(mean, 0)

        op = X
        expect = PauliExpectation(operator=op, backend=backend)
        mean = expect.compute_expectation(Plus)
        self.assertAlmostEqual(mean, 1)
        mean = expect.compute_expectation(Minus)
        self.assertAlmostEqual(mean, -1)
        mean = expect.compute_expectation(Plus+Minus)
        self.assertAlmostEqual(mean, 0)
