import unittest
from test.aqua import QiskitAquaTestCase
from qiskit import BasicAer
from qiskit.circuit.library import RealAmplitudes
from qiskit.aqua import QuantumInstance
from qiskit.aqua.operators import WeightedPauliOperator
from qiskit.aqua.algorithms import VQE
from qiskit.aqua import aqua_globals
from qiskit.aqua.components.optimizers import BOBYQA, SNOBFIT, IMFIL


class TestOptimizers(QiskitAquaTestCase):
    """ Test scikit-quant optimizers """

    def setUp(self):
        """ set the problem """
        super().setUp()
        aqua_globals.random_seed = 50
        pauli_dict = {
            'paulis': [{"coeff": {"imag": 0.0, "real": -1.052373245772859}, "label": "II"},
                       {"coeff": {"imag": 0.0, "real": 0.39793742484318045}, "label": "IZ"},
                       {"coeff": {"imag": 0.0, "real": -0.39793742484318045}, "label": "ZI"},
                       {"coeff": {"imag": 0.0, "real": -0.01128010425623538}, "label": "ZZ"},
                       {"coeff": {"imag": 0.0, "real": 0.18093119978423156}, "label": "XX"}
                       ]
        }
        self.qubit_op = WeightedPauliOperator.from_dict(pauli_dict)

    def _optimize(self, optimizer):
        """ launch vqe """
        result = VQE(self.qubit_op,
                     RealAmplitudes(),
                     optimizer).run(
                         QuantumInstance(BasicAer.get_backend('statevector_simulator'),
                                         seed_simulator=aqua_globals.random_seed,
                                         seed_transpiler=aqua_globals.random_seed))
        self.assertAlmostEqual(result.eigenvalue.real, -1.857, places=1)

    def test_bobyqa(self):
        """ bobyqa optimizer test """
        optimizer = BOBYQA(maxiter=150)
        self._optimize(optimizer)

    def test_snobfit(self):
        """ snobfit optimizer test """
        optimizer = SNOBFIT(maxiter=100, maxfail=100, maxmp=20)
        self._optimize(optimizer)

    def test_imfil(self):
        """ imfil test """
        optimizer = IMFIL(maxiter=100)
        self._optimize(optimizer)


if __name__ == '__main__':
    unittest.main()
