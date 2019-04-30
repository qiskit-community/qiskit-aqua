# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import unittest
import itertools
from parameterized import parameterized
from qiskit import BasicAer
from qiskit.aqua import QuantumInstance
from qiskit.aqua.components.oracles import TruthTableOracle
from qiskit.aqua.algorithms import DeutschJozsa
from test.common import QiskitAquaTestCase

bitmaps = ['0000', '0101', '1111', '11110000']
mct_modes = ['basic', 'basic-dirty-ancilla', 'advanced', 'noancilla']
optimizations = ['off', 'qm-dlx']
simulators = ['statevector_simulator', 'qasm_simulator']


class TestDeutschJozsa(QiskitAquaTestCase):
    @parameterized.expand(
        itertools.product(bitmaps, mct_modes, optimizations, simulators)
    )
    def test_deutsch_jozsa(self, dj_input, mct_mode, optimization, simulator):
        backend = BasicAer.get_backend(simulator)
        oracle = TruthTableOracle(dj_input, optimization=optimization, mct_mode=mct_mode)
        algorithm = DeutschJozsa(oracle)
        quantum_instance = QuantumInstance(backend)
        result = algorithm.run(quantum_instance=quantum_instance)
        # print(result['circuit'].draw(line_length=10000))
        if sum([int(i) for i in dj_input]) == len(dj_input) / 2:
            self.assertTrue(result['result'] == 'balanced')
        else:
            self.assertTrue(result['result'] == 'constant')


if __name__ == '__main__':
    unittest.main()
