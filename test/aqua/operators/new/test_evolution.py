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

from qiskit import QuantumCircuit, BasicAer
from qiskit.circuit import ParameterVector

from qiskit.aqua.operators import X, Y, Z, I, CX, T, H, S, OpPrimitive, OpSum, OpComposition, OpVec
from qiskit.aqua.operators import StateFn, Zero, One, Plus, Minus
from qiskit.aqua.operators import EvolutionBase, PauliTrotterEvolution


class TestEvolution(QiskitAquaTestCase):
    """Evolution tests."""

    def test_pauli_evolution(self):
        op = (2*Z^Z) + (3*X^X) - (4*Y^Y) + (.5*I^I)
        op = (-1.052373245772859 * I ^ I) + \
             (0.39793742484318045 * I ^ Z) + \
             (0.18093119978423156 * X ^ X) + \
             (-0.39793742484318045 * Z ^ I) + \
             (-0.01128010425623538 * Z ^ Z)
        backend = BasicAer.get_backend('qasm_simulator')
        evolution = EvolutionBase.factory(operator=op, backend=backend)
        # wf = (Pl^Pl) + (Ze^Ze)
        wf = ((np.pi/2)*op).exp_i() @ CX @ (H^I) @ Zero
        mean = evolution.convert(wf)
        self.assertIsNotNone(mean)
        print(mean.to_matrix())

    def test_parameterized_evolution(self):
        thetas = ParameterVector('θ', length=6)
        op = (thetas[0] * I ^ I) + \
             (thetas[1] * I ^ Z) + \
             (thetas[2] * X ^ X) + \
             (thetas[3] * Z ^ I) + \
             (thetas[4] * Y ^ Z) + \
             (thetas[5] * Z ^ Z)
        evolution = PauliTrotterEvolution(trotter_mode='trotter', reps=1, group_paulis=False)
        # wf = (Pl^Pl) + (Ze^Ze)
        wf = (op).exp_i() @ CX @ (H ^ I) @ Zero
        mean = evolution.convert(wf)
        circuit_params = mean.to_circuit().parameters
        # Check that the non-identity parameters are in the circuit
        for p in thetas[1:]:
            self.assertIn(p, circuit_params)
