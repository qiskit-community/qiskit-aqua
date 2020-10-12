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

import unittest

from test.chemistry import QiskitChemistryTestCase
from qiskit import BasicAer
from qiskit.aqua import QuantumInstance, aqua_globals
from qiskit.aqua.algorithms import VQE
from qiskit.aqua.components.optimizers import SLSQP, SPSA
from qiskit.aqua.operators import AerPauliExpectation, PauliExpectation
from qiskit.chemistry.components.initial_states import HartreeFock
from qiskit.chemistry.components.variational_forms import UCCSD
from qiskit.chemistry.core import QubitMappingType
from qiskit.chemistry.drivers import HDF5Driver
from qiskit.chemistry.ground_state_calculation import MinimumEigensolverGroundStateCalculation
from qiskit.chemistry.qubit_transformations import FermionicTransformation


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

        gsc = MinimumEigensolverGroundStateCalculation(self.fermionic_transformation, solver)

        result = gsc.compute_groundstate(self.driver)

        self.assertAlmostEqual(result.energy, self.reference_energy, places=6)

    def test_uccsd_hf_qasm(self):
        """ uccsd hf test with qasm_simulator. """
        backend = BasicAer.get_backend('qasm_simulator')
        optimizer = SPSA(maxiter=200, last_avg=5)
        solver = VQE(var_form=self.var_form, optimizer=optimizer,
                     expectation=PauliExpectation(),
                     quantum_instance=QuantumInstance(backend=backend,
                                                      seed_simulator=aqua_globals.random_seed,
                                                      seed_transpiler=aqua_globals.random_seed))

        gsc = MinimumEigensolverGroundStateCalculation(self.fermionic_transformation, solver)

        result = gsc.compute_groundstate(self.driver)

        self.assertAlmostEqual(result.energy, -1.138, places=2)

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

        gsc = MinimumEigensolverGroundStateCalculation(self.fermionic_transformation, solver)

        result = gsc.compute_groundstate(self.driver)

        self.assertAlmostEqual(result.energy, self.reference_energy, places=6)

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

        gsc = MinimumEigensolverGroundStateCalculation(self.fermionic_transformation, solver)

        result = gsc.compute_groundstate(self.driver)

        self.assertAlmostEqual(result.energy, -1.138, places=2)

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

        gsc = MinimumEigensolverGroundStateCalculation(self.fermionic_transformation, solver)

        result = gsc.compute_groundstate(self.driver)
        self.assertAlmostEqual(result.energy, self.reference_energy, places=3)


if __name__ == '__main__':
    unittest.main()
