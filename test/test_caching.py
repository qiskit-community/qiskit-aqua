import unittest

import numpy as np
from parameterized import parameterized
from qiskit_aqua import get_aer_backend

from test.common import QiskitAquaTestCase
from qiskit_aqua import Operator, run_algorithm, QuantumInstance
from qiskit_aqua.input import EnergyInput
from qiskit_aqua.components.variational_forms import RY
from qiskit_aqua.components.optimizers import L_BFGS_B
from qiskit_aqua.components.initial_states import Zero
from qiskit_aqua.algorithms.adaptive import VQE
from qiskit_aqua.utils import CircuitCache


class TestCaching(QiskitAquaTestCase):

    def setUp(self):
        super().setUp()
        np.random.seed(50)
        pauli_dict = {
            'paulis': [{"coeff": {"imag": 0.0, "real": -1.052373245772859}, "label": "II"},
                       {"coeff": {"imag": 0.0, "real": 0.39793742484318045}, "label": "IZ"},
                       {"coeff": {"imag": 0.0, "real": -0.39793742484318045}, "label": "ZI"},
                       {"coeff": {"imag": 0.0, "real": -0.01128010425623538}, "label": "ZZ"},
                       {"coeff": {"imag": 0.0, "real": 0.18093119978423156}, "label": "XX"}
                       ]
        }
        qubit_op = Operator.load_from_dict(pauli_dict)
        self.algo_input = EnergyInput(qubit_op)

    @parameterized.expand([
        ['statevector_simulator', 1],
        ['qasm_simulator', 1000]
    ])
    def test_vqe_caching_via_run_algorithm_sv(self, backend, shots):
        params_no_caching = {
            'algorithm': {'name': 'VQE'},
            'problem': {'name': 'energy',
                        'random_seed': 50,
                        'circuit_caching': False,
                        'caching_naughty_mode': False,
                        'circuit_cache_file': None,
                        },
            'backend': {'name': backend, 'shots': shots},
        }
        result_no_caching = run_algorithm(params_no_caching, self.algo_input)

        params_caching = {
            'algorithm': {'name': 'VQE'},
            'problem': {'name': 'energy',
                    'random_seed': 50,
                    'circuit_caching': True,
                    'caching_naughty_mode': False,
                    'circuit_cache_file': None,
                    },
            'backend': {'name': backend, 'shots': shots},
        }
        result_caching = run_algorithm(params_caching, self.algo_input)

        self.assertAlmostEqual(result_no_caching['energy'], result_caching['energy'])

        np.testing.assert_array_almost_equal(result_no_caching['eigvals'], result_caching['eigvals'], 5)
        np.testing.assert_array_almost_equal(result_no_caching['opt_params'], result_caching['opt_params'], 5)
        self.assertIn('eval_count', result_caching)
        self.assertIn('eval_time', result_caching)

    @parameterized.expand([
        ['statevector_simulator', 1],
        ['qasm_simulator', 1000]
    ])
    def test_vqe_caching_naughty_via_run_algorithm(self, backend, shots):
        params_caching = {
            'algorithm': {'name': 'VQE'},
            'problem': {'name': 'energy',
                        'random_seed': 50,
                        'circuit_caching': True,
                        'caching_naughty_mode': False,
                        'circuit_cache_file': None,
                        },
            'backend': {'name': backend, 'shots': shots},
        }
        result_caching = run_algorithm(params_caching, self.algo_input)

        params_caching_naughty = {
            'algorithm': {'name': 'VQE'},
            'problem': {'name': 'energy',
                        'random_seed': 50,
                        'circuit_caching': True,
                        'caching_naughty_mode': True,
                        'circuit_cache_file': None,
                        },
            'backend': {'name': backend, 'shots': shots},
        }
        result_caching_naughty = run_algorithm(params_caching_naughty, self.algo_input)

        self.assertAlmostEqual(result_caching['energy'], result_caching_naughty['energy'])

        np.testing.assert_array_almost_equal(result_caching_naughty['eigvals'], result_caching['eigvals'], 5)
        np.testing.assert_array_almost_equal(result_caching_naughty['opt_params'], result_caching['opt_params'], 5)
        self.assertIn('eval_count', result_caching_naughty)
        self.assertIn('eval_time', result_caching_naughty)

    # @parameterized.expand([
    #     [True],
    #     [False]
    # ])
    def test_vqe_caching_direct(self, batch_mode=False):
        backend = get_aer_backend('statevector_simulator')
        num_qubits = self.algo_input.qubit_op.num_qubits
        init_state = Zero(num_qubits)
        var_form = RY(num_qubits, 3, initial_state=init_state)
        optimizer = L_BFGS_B()
        algo = VQE(self.algo_input.qubit_op, var_form, optimizer, 'matrix', batch_mode=batch_mode)
        quantum_instance = QuantumInstance(backend)
        circuit_cache = CircuitCache(caching_naughty_mode=True)
        quantum_instance_caching = QuantumInstance(backend, circuit_cache=circuit_cache)
        result_no_caching = algo.run(quantum_instance)
        result_caching = algo.run(quantum_instance_caching)
        self.assertAlmostEqual(result_no_caching['energy'], result_caching['energy'])

if __name__ == '__main__':
    unittest.main()