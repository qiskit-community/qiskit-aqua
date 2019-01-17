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

        backends = ['statevector_simulator', 'qasm_simulator']
        res = {}
        for backend in backends:
            params_no_caching = {
                'algorithm': {'name': 'VQE'},
                'problem': {'name': 'energy',
                            'random_seed': 50,
                            'circuit_caching': False,
                            'skip_qobj_deepcopy': False,
                            'skip_qobj_validation': False,
                            'circuit_cache_file': None,
                            },
                'backend': {'name': backend, 'shots': 1000},
            }
            res[backend] = run_algorithm(params_no_caching, self.algo_input)
        self.reference_vqe_result = res

    @parameterized.expand([
        ['statevector_simulator', True, True, True],
        ['qasm_simulator', True, True, True],
        ['statevector_simulator', True, False, False],
        ['statevector_simulator', False, False, True],
    ])
    def test_vqe_caching_via_run_algorithm(self, backend, caching, skip_qobj_deepcopy, skip_validation):
        params_caching = {
            'algorithm': {'name': 'VQE'},
            'problem': {'name': 'energy',
                        'random_seed': 50,
                        'circuit_caching': caching,
                        'skip_qobj_deepcopy': skip_qobj_deepcopy,
                        'skip_qobj_validation': skip_validation,
                        'circuit_cache_file': None,
                        },
            'backend': {'name': backend, 'shots': 1000},
        }
        result_caching = run_algorithm(params_caching, self.algo_input)

        self.assertAlmostEqual(result_caching['energy'], self.reference_vqe_result[backend]['energy'])

        np.testing.assert_array_almost_equal(self.reference_vqe_result[backend]['eigvals'], result_caching['eigvals'], 5)
        np.testing.assert_array_almost_equal(self.reference_vqe_result[backend]['opt_params'], result_caching['opt_params'], 5)
        self.assertIn('eval_count', result_caching)
        self.assertIn('eval_time', result_caching)

    @parameterized.expand([
        [True],
        [False]
    ])
    def test_vqe_caching_direct(self, batch_mode):
        backend = get_aer_backend('statevector_simulator')
        num_qubits = self.algo_input.qubit_op.num_qubits
        init_state = Zero(num_qubits)
        var_form = RY(num_qubits, 3, initial_state=init_state)
        optimizer = L_BFGS_B()
        algo = VQE(self.algo_input.qubit_op, var_form, optimizer, 'matrix', batch_mode=batch_mode)
        circuit_cache = CircuitCache(skip_qobj_deepcopy=True)
        quantum_instance_caching = QuantumInstance(backend, circuit_cache=circuit_cache, skip_qobj_validation=True)
        result_caching = algo.run(quantum_instance_caching)
        self.assertAlmostEqual(self.reference_vqe_result['statevector_simulator']['energy'], result_caching['energy'])

if __name__ == '__main__':
    unittest.main()