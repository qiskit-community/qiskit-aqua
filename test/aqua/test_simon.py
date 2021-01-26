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

""" Test Simon """

import unittest
import math
import itertools
from test.aqua import QiskitAquaTestCase
import numpy as np
from ddt import ddt, idata, unpack
from qiskit.aqua import QuantumInstance
from qiskit.aqua.components.oracles import TruthTableOracle
from qiskit.aqua.algorithms import Simon
from qiskit import BasicAer

BITMAPS = [
    ['00011110', '01100110', '10101010'],
    ['10010110', '01010101', '10000010'],
    ['01101001', '10011001', '01100110'],
]
MCT_MODES = ['basic', 'basic-dirty-ancilla', 'advanced', 'noancilla']
OPTIMIZATIONS = [True, False]
SIMULATORS = ['statevector_simulator', 'qasm_simulator']


@ddt
class TestSimon(QiskitAquaTestCase):
    """ Test Simon """
    @idata(itertools.product(BITMAPS, MCT_MODES, OPTIMIZATIONS, SIMULATORS))
    @unpack
    def test_simon(self, simon_input, mct_mode, optimization, simulator):
        """ Simon test """
        # find the two keys that have matching values
        nbits = int(math.log(len(simon_input[0]), 2))
        vals = list(zip(*simon_input))[::-1]

        def find_pair():
            for i, val in enumerate(vals):
                for j in range(i + 1, len(vals)):
                    if val == vals[j]:
                        return i, j
            return 0, 0

        k_1, k_2 = find_pair()
        hidden = np.binary_repr(k_1 ^ k_2, nbits)

        backend = BasicAer.get_backend(simulator)
        oracle = TruthTableOracle(simon_input, optimization=optimization, mct_mode=mct_mode)
        algorithm = Simon(oracle)
        quantum_instance = QuantumInstance(backend)
        result = algorithm.run(quantum_instance=quantum_instance)
        # print(result['circuit'].draw(line_length=10000))
        self.assertEqual(result['result'], hidden)


if __name__ == '__main__':
    unittest.main()
