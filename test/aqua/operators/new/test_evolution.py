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

from qiskit.aqua.operators import X, Y, Z, I, CX, T, H, S, OpPrimitive, OpSum, OpComposition, OpVec
from qiskit.aqua.operators import StateFn, Zero, One, Plus, Minus

from qiskit.aqua.operators import EvolutionBase
from qiskit import QuantumCircuit, BasicAer


class TestEvolution(QiskitAquaTestCase):
    """Evolution tests."""

    def test_pauli_evolution(self):
        op = (2*Z^Z) + (3*X^X) - (4*Y^Y) + (.5*I^I)
        backend = BasicAer.get_backend('qasm_simulator')
        evolution = EvolutionBase.factory(operator=op, backend=backend)
        # wf = (Pl^Pl) + (Ze^Ze)
        wf = (np.pi/2)*op.exp_i() @ CX @ (H^I) @ Zero
        mean = evolution.convert(wf)
        print(mean)
