import unittest

import numpy as np
from qiskit.aqua.components.variational_forms import VariationalForm, RY
from qiskit import transpile
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import CXCancellation

from qiskit.aqua.components.extrapolation_pass_managers import RedundantCNOT
from qiskit.aqua.components.variational_forms import ExtrapolatedVF

from test.aqua.common import QiskitAquaTestCase

class VariationalWrapperTest(unittest.TestCase):
    def setUp(self):
        self.cnot_pm = PassManager(passes=CXCancellation())

    def test_basic_2(self):
        self.vf = RY(num_qubits=2)
        pm = PassManager(passes=[RedundantCNOT(redundant_pairs=1)])
        wrapped_vf = ExtrapolatedVF(
            variational_form=self.vf,
            pass_manager=pm
        )
        pars = np.random.uniform(0, 2*np.pi, size=wrapped_vf.num_parameters)
        circ = wrapped_vf.construct_circuit(parameters=pars)

        self.assertEqual(
            transpile(circ, pass_manager=self.cnot_pm),
            self.vf.construct_circuit(parameters=pars)
        )

    def test_basic_3(self):
        self.vf = RY(num_qubits=3)
        pm = PassManager(passes=[RedundantCNOT(redundant_pairs=1)])
        wrapped_vf = ExtrapolatedVF(
            variational_form=self.vf,
            pass_manager=pm
        )
        pars = np.random.uniform(0, 2*np.pi, size=wrapped_vf.num_parameters)
        circ = wrapped_vf.construct_circuit(parameters=pars)

        self.assertEqual(
            transpile(circ, pass_manager=self.cnot_pm),
            self.vf.construct_circuit(parameters=pars)
        )


if __name__ == '__main__':
    unittest.main()
