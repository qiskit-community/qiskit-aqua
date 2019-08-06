import logging
import unittest
import warnings

from qiskit import IBMQ
from qiskit.providers.aer import Aer
from qiskit.providers.aer.noise.device import basic_device_noise_model
from qiskit.quantum_info import Pauli
from qiskit.transpiler import PassManager
from scipy import interpolate

from qiskit.aqua import Operator
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import VQE, ExactEigensolver
from qiskit.aqua.components.extrapolation_pass_managers import RedundantCNOT
from qiskit.aqua.components.extrapolation_pass_managers import RichardsonExtrapolator
from qiskit.aqua.components.optimizers import DIRECT_L
from qiskit.aqua.components.variational_forms import RY

IBMQ.load_accounts()
logger = logging.getLogger(__name__)
warnings.filterwarnings('once')


class RichardsonExtrapolatorTest(unittest.TestCase):

    def setUp(self):
        self.extrapolator = lambda x, y, val: interpolate.interp1d(x, y, fill_value='extrapolate', kind='linear')(val)
        self.order_parameters = [1, 3, 5]
        self.extrapolated_value = 0
        self.pass_mans = [PassManager(RedundantCNOT(i)) for i in self.order_parameters]

        self.richardson_extrapolator = RichardsonExtrapolator(
            self.extrapolator,
            self.pass_mans,
            self.order_parameters,
            self.extrapolated_value
        )

    def test_extrapolation(self):
        energies = [3*x+2 for x in self.order_parameters]

        result = self.richardson_extrapolator.extrapolate(energies)

        self.assertAlmostEqual(
            2.0,
            result,
            places=5,
            msg='Found interpolated value {}, expected {}'.format(result, 2.0)
        )

    def test_simple_vqe(self):
        qubit_op = Operator(paulis=[
            [1.0, Pauli(z=[0, 0], x=[1, 1])],
            [0.5, Pauli(z=[1, 0], x=[0, 1])]
        ])
        ansatz = RY(2, depth=3)
        optimizer = DIRECT_L(max_evals=200)

        exact_energy = ExactEigensolver(qubit_op).run()['eigvals'][0].real

        vqe = VQE(
            operator=qubit_op,
            var_form=ansatz,
            optimizer=optimizer,
            richardson_extrapolator=self.richardson_extrapolator
        )

        backend = Aer.get_backend('qasm_simulator')
        properties = IBMQ.get_backend('ibmq_poughkeepsie').properties()
        noise_model = basic_device_noise_model(properties)
        qi = QuantumInstance(backend=backend, seed_simulator=42, optimization_level=0, shots=128,
                             noise_model=noise_model)

        result = vqe.run(quantum_instance=qi)

        self.assertAlmostEqual(
            result['eigvals'][0].real,
            exact_energy,
            places=2,
            msg='Found energy from extrapolated VQE {}, whereas the exact result is {}'.format(
                result['eigvals'][0].real,
                exact_energy
            )
        )


if __name__ == '__main__':
    unittest.main()
