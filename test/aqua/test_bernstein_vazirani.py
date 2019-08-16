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

""" Test Bernstein Vazirani """

import unittest
import itertools
import math
from test.aqua.common import QiskitAquaTestCase
from parameterized import parameterized
from qiskit import BasicAer
from qiskit.aqua import QuantumInstance
from qiskit.aqua.components.oracles import TruthTableOracle
from qiskit.aqua.algorithms import BernsteinVazirani

BITMAPS = ['00111100', '01011010']
MCT_MODES = ['basic', 'basic-dirty-ancilla', 'advanced', 'noancilla']
OPTIMIZATIONS = [True, False]
SIMULATORS = ['statevector_simulator', 'qasm_simulator']


class TestBernsteinVazirani(QiskitAquaTestCase):
    """ Test Berstein Vazirani """
    @parameterized.expand(
        itertools.product(BITMAPS, MCT_MODES, OPTIMIZATIONS, SIMULATORS)
    )
    def test_bernstein_vazirani(self, bv_input, mct_mode, optimization, simulator):
        """ Berstein Vazirani test """
        nbits = int(math.log(len(bv_input), 2))
        # compute the ground-truth classically
        parameter = ""
        for i in reversed(range(nbits)):
            bit = bv_input[2 ** i]
            parameter += bit

        backend = BasicAer.get_backend(simulator)
        oracle = TruthTableOracle(bv_input, optimization=optimization, mct_mode=mct_mode)
        algorithm = BernsteinVazirani(oracle)
        quantum_instance = QuantumInstance(backend)
        result = algorithm.run(quantum_instance=quantum_instance)
        # print(result['circuit'].draw(line_length=10000))
        self.assertEqual(result['result'], parameter)


if __name__ == '__main__':
    unittest.main()
