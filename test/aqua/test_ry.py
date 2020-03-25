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

""" Test RYCRX """

import unittest
from test.aqua import QiskitAquaTestCase

from ddt import ddt, idata, unpack
from qiskit import BasicAer

from qiskit.aqua import aqua_globals, QuantumInstance
from qiskit.aqua.algorithms import VQE
from qiskit.aqua.components.variational_forms import RY
from qiskit.aqua.components.optimizers import L_BFGS_B
from qiskit.aqua.operators import WeightedPauliOperator


@ddt
class TestRYCRX(QiskitAquaTestCase):
    """ Test RYCRX """

    def setUp(self):
        super().setUp()
        self.seed = 99
        aqua_globals.random_seed = self.seed
        pauli_dict = {
            'paulis': [{"coeff": {"imag": 0.0, "real": -1.052373245772859}, "label": "II"},
                       {"coeff": {"imag": 0.0, "real": 0.39793742484318045}, "label": "IZ"},
                       {"coeff": {"imag": 0.0, "real": -0.39793742484318045}, "label": "ZI"},
                       {"coeff": {"imag": 0.0, "real": -0.01128010425623538}, "label": "ZZ"},
                       {"coeff": {"imag": 0.0, "real": 0.18093119978423156}, "label": "XX"}
                       ]
        }
        self.qubit_op = WeightedPauliOperator.from_dict(pauli_dict)

    @idata([
        [2, 5],
        [3, 5],
        [4, 5]
    ])
    @unpack
    def test_vqe_var_forms(self, depth, places):
        """ VQE Var Forms test """
        aqua_globals.random_seed = self.seed
        result = VQE(self.qubit_op,
                     RY(self.qubit_op.num_qubits,
                        depth=depth, entanglement='sca',
                        entanglement_gate='crx', skip_final_ry=True),
                     L_BFGS_B()).run(QuantumInstance(BasicAer.get_backend('statevector_simulator'),
                                                     shots=1,
                                                     seed_simulator=aqua_globals.random_seed,
                                                     seed_transpiler=aqua_globals.random_seed))
        self.assertAlmostEqual(result.eigenvalue.real, -1.85727503, places=places)


if __name__ == '__main__':
    unittest.main()
