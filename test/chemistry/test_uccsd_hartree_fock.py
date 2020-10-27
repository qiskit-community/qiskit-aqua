# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test of UCCSD and HartreeFock Aqua extensions """

from test.chemistry import QiskitChemistryTestCase
from ddt import ddt, idata, unpack
from qiskit import BasicAer
from qiskit.aqua import QuantumInstance, aqua_globals
from qiskit.aqua.algorithms import VQE
from qiskit.aqua.components.optimizers import SLSQP, SPSA
from qiskit.aqua.operators import AerPauliExpectation, PauliExpectation

from qiskit.chemistry.circuit.library import HartreeFock
from qiskit.chemistry.components.variational_forms import UCCSD
from qiskit.chemistry.core import QubitMappingType
from qiskit.chemistry.drivers import HDF5Driver
from qiskit.chemistry.algorithms.ground_state_solvers import GroundStateEigensolver
from qiskit.chemistry.transformations import FermionicTransformation


@ddt
class TestUCCSDHartreeFock(QiskitChemistryTestCase):
    """Test for these aqua extensions."""

    def setUp(self):
        super().setUp()
        self.reference_energy = -1.1373060356951838

        self.seed = 700
        aqua_globals.random_seed = self.seed

        self.driver = HDF5Driver(self.get_resource_path('test_driver_hdf5.hdf5'))
        fermionic_transformation = FermionicTransformation(qubit_mapping=QubitMappingType.PARITY,
                                                           two_qubit_reduction=False)

        self.qubit_op, _ = fermionic_transformation.transform(self.driver)
        self.fermionic_transformation = fermionic_transformation

        self.optimizer = SLSQP(maxiter=100)
        initial_state = HartreeFock(
            fermionic_transformation.molecule_info['num_orbitals'],
            fermionic_transformation.molecule_info['num_particles'],
            qubit_mapping=fermionic_transformation._qubit_mapping,
            two_qubit_reduction=fermionic_transformation._two_qubit_reduction)
        self.var_form = UCCSD(
            num_orbitals=fermionic_transformation.molecule_info['num_orbitals'],
            num_particles=fermionic_transformation.molecule_info['num_particles'],
            initial_state=initial_state,
            qubit_mapping=fermionic_transformation._qubit_mapping,
            two_qubit_reduction=fermionic_transformation._two_qubit_reduction)

    def test_uccsd_hf(self):
        """ uccsd hf test """
        backend = BasicAer.get_backend('statevector_simulator')
        solver = VQE(var_form=self.var_form, optimizer=self.optimizer,
                     quantum_instance=QuantumInstance(backend=backend))

        gsc = GroundStateEigensolver(self.fermionic_transformation, solver)

        result = gsc.solve(self.driver)

        self.assertAlmostEqual(result.total_energies[0], self.reference_energy, places=6)

    def test_uccsd_hf_qasm(self):
        """ uccsd hf test with qasm_simulator. """
        backend = BasicAer.get_backend('qasm_simulator')
        optimizer = SPSA(maxiter=200, last_avg=5)
        solver = VQE(var_form=self.var_form, optimizer=optimizer,
                     expectation=PauliExpectation(),
                     quantum_instance=QuantumInstance(backend=backend,
                                                      seed_simulator=aqua_globals.random_seed,
                                                      seed_transpiler=aqua_globals.random_seed))

        gsc = GroundStateEigensolver(self.fermionic_transformation, solver)

        result = gsc.solve(self.driver)
        self.assertAlmostEqual(result.total_energies[0], -1.138, places=2)

    def test_uccsd_hf_aer_statevector(self):
        """ uccsd hf test with Aer statevector """
        try:
            # pylint: disable=import-outside-toplevel
            from qiskit import Aer
        except Exception as ex:  # pylint: disable=broad-except
            self.skipTest("Aer doesn't appear to be installed. Error: '{}'".format(str(ex)))
            return
        backend = Aer.get_backend('statevector_simulator')
        solver = VQE(var_form=self.var_form, optimizer=self.optimizer,
                     quantum_instance=QuantumInstance(backend=backend))

        gsc = GroundStateEigensolver(self.fermionic_transformation, solver)

        result = gsc.solve(self.driver)
        self.assertAlmostEqual(result.total_energies[0], self.reference_energy, places=6)

    def test_uccsd_hf_aer_qasm(self):
        """ uccsd hf test with Aer qasm_simulator. """
        try:
            # pylint: disable=import-outside-toplevel
            from qiskit import Aer
        except Exception as ex:  # pylint: disable=broad-except
            self.skipTest("Aer doesn't appear to be installed. Error: '{}'".format(str(ex)))
            return
        backend = Aer.get_backend('qasm_simulator')
        optimizer = SPSA(maxiter=200, last_avg=5)
        solver = VQE(var_form=self.var_form, optimizer=optimizer,
                     expectation=PauliExpectation(),
                     quantum_instance=QuantumInstance(backend=backend,
                                                      seed_simulator=aqua_globals.random_seed,
                                                      seed_transpiler=aqua_globals.random_seed))

        gsc = GroundStateEigensolver(self.fermionic_transformation, solver)

        result = gsc.solve(self.driver)
        self.assertAlmostEqual(result.total_energies[0], -1.138, places=2)

    def test_uccsd_hf_aer_qasm_snapshot(self):
        """ uccsd hf test with Aer qasm_simulator snapshot. """
        try:
            # pylint: disable=import-outside-toplevel
            from qiskit import Aer
        except Exception as ex:  # pylint: disable=broad-except
            self.skipTest("Aer doesn't appear to be installed. Error: '{}'".format(str(ex)))
            return
        backend = Aer.get_backend('qasm_simulator')
        optimizer = SPSA(maxiter=200, last_avg=5)
        solver = VQE(var_form=self.var_form, optimizer=optimizer,
                     expectation=AerPauliExpectation(),
                     quantum_instance=QuantumInstance(backend=backend))

        gsc = GroundStateEigensolver(self.fermionic_transformation, solver)

        result = gsc.solve(self.driver)
        self.assertAlmostEqual(result.total_energies[0], self.reference_energy, places=3)

    EXCITATION_RESULTS = \
        [[[[0, 1], [0, 2], [3, 4], [3, 5]],
          [[0, 1, 3, 4], [0, 1, 3, 5], [0, 2, 3, 4], [0, 2, 3, 5]]],  # 0 full: 6 orbs, 2 particles
         [[[0, 2], [3, 5]], [[0, 2, 3, 5]]],  # 1 limited active space
         [[[0, 1], [0, 2], [3, 4], [3, 5]], []],  # 2 singles only
         [[], [[0, 1, 3, 4], [0, 1, 3, 5], [0, 2, 3, 4], [0, 2, 3, 5]]],  # 3 doubles only
         [[[0, 1], [3, 4]], []],  # 4 singles only limited active space
         [[[0, 2], [1, 2], [3, 5], [4, 5]],
          [[0, 2, 3, 5], [0, 2, 4, 5], [1, 2, 3, 5], [1, 2, 4, 5]]],  # 5 full: 6 orbs, 4 particles
         [[[1, 2], [4, 5]], [[1, 2, 4, 5]]],  # 6 limited active space
         [[[0, 2], [0, 3], [1, 2], [1, 3], [4, 6], [4, 7], [5, 6], [5, 7]],  # 7
          [[0, 2, 4, 6], [0, 2, 4, 7], [0, 2, 5, 6], [0, 2, 5, 7], [0, 3, 4, 6], [0, 3, 4, 7],
           [0, 3, 5, 6], [0, 3, 5, 7], [1, 2, 4, 6], [1, 2, 4, 7], [1, 2, 5, 6], [1, 2, 5, 7],
           [1, 3, 4, 6], [1, 3, 4, 7], [1, 3, 5, 6], [1, 3, 5, 7], [0, 2, 1, 3], [4, 6, 5, 7]]],
         [[[0, 2], [0, 3], [1, 2], [1, 3], [4, 6], [4, 7], [5, 6], [5, 7]],  # 8 No same spins
          [[0, 2, 4, 6], [0, 2, 4, 7], [0, 2, 5, 6], [0, 2, 5, 7], [0, 3, 4, 6], [0, 3, 4, 7],
           [0, 3, 5, 6], [0, 3, 5, 7], [1, 2, 4, 6], [1, 2, 4, 7], [1, 2, 5, 6], [1, 2, 5, 7],
           [1, 3, 4, 6], [1, 3, 4, 7], [1, 3, 5, 6], [1, 3, 5, 7]]],
         ]

    @idata([[0, 6, 2],
            [0, 6, 2, [0], [0, 1]],  # Full active space
            [1, 6, 2, [0], [1]],     # Restrict active space
            [0, 6, 2, [0], [0, 1], False],
            [2, 6, 2, None, None, True, 'both', 'ucc', 's'],
            [3, 6, 2, None, [0, 1], True, 'both', 'ucc', 'd'],
            [4, 6, 2, [0], [0], False, 'both', 'ucc', 's'],
            [5, 6, 4],
            [5, 6, 4, [0, 1], [0]],  # Full active space
            [6, 6, 4, [1], [0]],     # Restrict active space
            [7, 8, 4],
            [8, 8, 4, None, None, False],
            ])
    @unpack
    def test_uccsd_excitations(self, expected_result_idx, num_orbitals, num_particles,
                               active_occupied=None, active_unoccupied=None,
                               same_spin_doubles=True,
                               method_singles='both', method_doubles='ucc',
                               excitation_type='sd'
                               ):
        """ Test generated excitation lists in conjunction with active space """

        excitations = UCCSD.compute_excitation_lists(
            num_orbitals=num_orbitals, num_particles=num_particles,
            active_occ_list=active_occupied, active_unocc_list=active_unoccupied,
            same_spin_doubles=same_spin_doubles,
            method_singles=method_singles, method_doubles=method_doubles,
            excitation_type=excitation_type)

        self.assertListEqual(list(excitations), self.EXCITATION_RESULTS[expected_result_idx])
