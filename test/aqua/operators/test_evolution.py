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

from qiskit.circuit import ParameterVector

from qiskit.aqua.operators import (X, Y, Z, I, CX, H, ListOp, CircuitOp, Zero, EvolutionBase,
                                   EvolvedOp, PauliTrotterEvolution, QDrift)


# pylint: disable=invalid-name

class TestEvolution(QiskitAquaTestCase):
    """Evolution tests."""

    def test_pauli_evolution(self):
        """ pauli evolution test """
        op = (2 * Z ^ Z) + (3 * X ^ X) - (4 * Y ^ Y) + (.5 * I ^ I)
        op = (-1.052373245772859 * I ^ I) + \
             (0.39793742484318045 * I ^ Z) + \
             (0.18093119978423156 * X ^ X) + \
             (-0.39793742484318045 * Z ^ I) + \
             (-0.01128010425623538 * Z ^ Z)
        evolution = EvolutionBase.factory(operator=op)
        # wf = (Pl^Pl) + (Ze^Ze)
        wf = ((np.pi / 2) * op).exp_i() @ CX @ (H ^ I) @ Zero
        mean = evolution.convert(wf)
        self.assertIsNotNone(mean)
        # print(mean.to_matrix())

    def test_parameterized_evolution(self):
        """ parameterized evolution test """
        thetas = ParameterVector('θ', length=7)
        op = (thetas[0] * I ^ I) + \
             (thetas[1] * I ^ Z) + \
             (thetas[2] * X ^ X) + \
             (thetas[3] * Z ^ I) + \
             (thetas[4] * Y ^ Z) + \
             (thetas[5] * Z ^ Z)
        op = op * thetas[6]
        evolution = PauliTrotterEvolution(trotter_mode='trotter', reps=1, group_paulis=False)
        # wf = (Pl^Pl) + (Ze^Ze)
        wf = (op).exp_i() @ CX @ (H ^ I) @ Zero
        mean = evolution.convert(wf)
        circuit_params = mean.to_circuit().parameters
        # Check that the non-identity parameters are in the circuit
        for p in thetas[1:]:
            self.assertIn(p, circuit_params)
        self.assertNotIn(thetas[0], circuit_params)

    def test_bind_parameters(self):
        """ bind parameters test """
        thetas = ParameterVector('θ', length=6)
        op = (thetas[1] * I ^ Z) + \
             (thetas[2] * X ^ X) + \
             (thetas[3] * Z ^ I) + \
             (thetas[4] * Y ^ Z) + \
             (thetas[5] * Z ^ Z)
        op = thetas[0] * op
        evolution = PauliTrotterEvolution(trotter_mode='trotter', reps=1, group_paulis=False)
        # wf = (Pl^Pl) + (Ze^Ze)
        wf = (op).exp_i() @ CX @ (H ^ I) @ Zero
        wf = wf.bind_parameters({thetas: np.arange(10, 16)})
        mean = evolution.convert(wf)
        circuit_params = mean.to_circuit().parameters
        # Check that the no parameters are in the circuit
        for p in thetas[1:]:
            self.assertNotIn(p, circuit_params)

    def test_bind_circuit_parameters(self):
        """ bind circuit parameters test """
        thetas = ParameterVector('θ', length=6)
        op = (thetas[1] * I ^ Z) + \
             (thetas[2] * X ^ X) + \
             (thetas[3] * Z ^ I) + \
             (thetas[4] * Y ^ Z) + \
             (thetas[5] * Z ^ Z)
        op = thetas[0] * op
        evolution = PauliTrotterEvolution(trotter_mode='trotter', reps=1, group_paulis=False)
        # wf = (Pl^Pl) + (Ze^Ze)
        wf = (op).exp_i() @ CX @ (H ^ I) @ Zero
        evo = evolution.convert(wf)
        mean = evo.bind_parameters({thetas: np.arange(10, 16)})
        # Check that the no parameters are in the circuit
        for p in thetas[1:]:
            self.assertNotIn(p, mean.to_circuit().parameters)
        # Check that original circuit is unchanged
        for p in thetas:
            self.assertIn(p, evo.to_circuit().parameters)

    # TODO test with other Op types than CircuitStateFn
    def test_bind_parameter_list(self):
        """ bind parameters list test """
        thetas = ParameterVector('θ', length=6)
        op = (thetas[1] * I ^ Z) + \
             (thetas[2] * X ^ X) + \
             (thetas[3] * Z ^ I) + \
             (thetas[4] * Y ^ Z) + \
             (thetas[5] * Z ^ Z)
        op = thetas[0] * op
        evolution = PauliTrotterEvolution(trotter_mode='trotter', reps=1, group_paulis=False)
        # wf = (Pl^Pl) + (Ze^Ze)
        wf = (op).exp_i() @ CX @ (H ^ I) @ Zero
        evo = evolution.convert(wf)
        param_list = np.transpose([np.arange(10, 16), np.arange(2, 8), np.arange(30, 36)]).tolist()
        means = evo.bind_parameters({thetas: param_list})
        self.assertIsInstance(means, ListOp)
        # Check that the no parameters are in the circuit
        for p in thetas[1:]:
            for circop in means.oplist:
                self.assertNotIn(p, circop.to_circuit().parameters)
        # Check that original circuit is unchanged
        for p in thetas:
            self.assertIn(p, evo.to_circuit().parameters)

    def test_qdrift(self):
        """ QDrift test """
        op = (2 * Z ^ Z) + (3 * X ^ X) - (4 * Y ^ Y) + (.5 * Z ^ I)
        trotterization = QDrift().trotterize(op)
        self.assertGreater(len(trotterization.oplist), 150)
        last_coeff = None
        # Check that all types are correct and all coefficients are equals
        for op in trotterization.oplist:
            self.assertIsInstance(op, (EvolvedOp, CircuitOp))
            if isinstance(op, EvolvedOp):
                if last_coeff:
                    self.assertEqual(op.primitive.coeff, last_coeff)
                else:
                    last_coeff = op.primitive.coeff
