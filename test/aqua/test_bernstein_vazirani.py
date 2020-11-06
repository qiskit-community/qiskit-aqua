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

""" Test Bernstein Vazirani """

import unittest
import itertools
import math
from test.aqua import QiskitAquaTestCase
from ddt import ddt, idata, unpack
from qiskit import BasicAer
from qiskit.transpiler import PassManagerConfig
from qiskit.transpiler.preset_passmanagers import level_0_pass_manager
from qiskit.aqua import QuantumInstance
from qiskit.aqua.components.oracles import TruthTableOracle
from qiskit.aqua.algorithms import BernsteinVazirani

BITMAPS = ['00111100', '01011010']
MCT_MODES = ['basic', 'basic-dirty-ancilla', 'advanced', 'noancilla']
OPTIMIZATIONS = [True, False]
SIMULATORS = ['statevector_simulator', 'qasm_simulator']


@ddt
class TestBernsteinVazirani(QiskitAquaTestCase):
    """ Test Bernstein Vazirani """
    @idata(itertools.product(BITMAPS, MCT_MODES, OPTIMIZATIONS, SIMULATORS))
    @unpack
    def test_bernstein_vazirani(self, bv_input, mct_mode, optimization, simulator):
        """ Test Bernstein Vazirani """
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

    def test_with_pass_manager(self):
        """ Test Bernstein Vazirani using PassManager """
        quantum_instance = QuantumInstance(
            BasicAer.get_backend('qasm_simulator'),
            pass_manager=level_0_pass_manager(
                PassManagerConfig(basis_gates=['cx', 'u1', 'u2', 'u3'])))
        alg = BernsteinVazirani(oracle=TruthTableOracle(bitmaps="01100110"),
                                quantum_instance=quantum_instance)
        result = alg.run()
        self.assertEqual(result['result'], '011')


if __name__ == '__main__':
    unittest.main()
