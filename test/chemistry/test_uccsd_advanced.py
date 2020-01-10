# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test of UCCSD and HartreeFock Aqua extensions """

from test.chemistry.common import QiskitChemistryTestCase
from qiskit import BasicAer
from qiskit.aqua import QuantumInstance
from qiskit.aqua.operators import Z2Symmetries
from qiskit.aqua.algorithms import VQE, ExactEigensolver
from qiskit.aqua.components.optimizers import SLSQP
from qiskit.chemistry.components.initial_states import HartreeFock
from qiskit.chemistry.components.variational_forms import UCCSD
from qiskit.chemistry.drivers import PySCFDriver, UnitsType
from qiskit.chemistry.core import Hamiltonian, QubitMappingType, TransformationType

# pylint: disable=invalid-name


class TestUCCSDHartreeFock(QiskitChemistryTestCase):
    """Test for these aqua extensions."""

    def setUp(self):
        super().setUp()
        self.molecule = "H 0.000000 0.000000 0.735000;H 0.000000 0.000000 0.000000"
        self.driver = PySCFDriver(atom=self.molecule,
                                  unit=UnitsType.ANGSTROM,
                                  charge=0,
                                  spin=0,
                                  basis='631g')
        self.qmolecule = self.driver.run()
        self.core = Hamiltonian(transformation=TransformationType.FULL,
                                qubit_mapping=QubitMappingType.PARITY,
                                two_qubit_reduction=True,
                                freeze_core=True,
                                orbital_reduction=[])
        self.qubit_op, _ = self.core.run(self.qmolecule)

        z2_symmetries = Z2Symmetries.find_Z2_symmetries(self.qubit_op)
        tapered_ops = z2_symmetries.taper(self.qubit_op)
        smallest_eig_value = 99999999999999
        smallest_idx = -1
        for idx, _ in enumerate(tapered_ops):
            ee = ExactEigensolver(tapered_ops[idx], k=1)
            curr_value = ee.run()['energy']
            if curr_value < smallest_eig_value:
                smallest_eig_value = curr_value
                smallest_idx = idx
        self.the_tapered_op = tapered_ops[smallest_idx]

        self.reference_energy_pUCCD = -1.1434447924298028
        self.reference_energy_UCCD0 = -1.1476045878481704
        self.reference_energy_UCCD0full = -1.1515491334334347
        # reference energy of UCCSD/VQE with tapering everywhere
        self.reference_energy_UCCSD = -1.1516142309717594
        # reference energy of UCCSD/VQE when no tapering on excitations is used
        self.reference_energy_UCCSD_no_tap_exc = -1.1516142309717594
        # excitations for succ
        self.reference_singlet_double_excitations = [[0, 1, 4, 5], [0, 1, 4, 6], [0, 1, 4, 7],
                                                     [0, 2, 4, 6], [0, 2, 4, 7], [0, 3, 4, 7]]
        # groups for succ_full
        self.reference_singlet_groups = [[[0, 1, 4, 5]], [[0, 1, 4, 6], [0, 2, 4, 5]],
                                         [[0, 1, 4, 7], [0, 3, 4, 5]], [[0, 2, 4, 6]],
                                         [[0, 2, 4, 7], [0, 3, 4, 6]], [[0, 3, 4, 7]]]
        pass

    def test_uccsd_hf_qpUCCD(self):
        """ paired uccd test """

        optimizer = SLSQP(maxiter=100)
        initial_state = HartreeFock(self.qubit_op.num_qubits,
                                    self.core.molecule_info['num_orbitals'],
                                    self.core.molecule_info['num_particles'],
                                    qubit_mapping=self.core._qubit_mapping,
                                    two_qubit_reduction=self.core._two_qubit_reduction)

        var_form = UCCSD(num_qubits=self.qubit_op.num_qubits, depth=1,
                         num_orbitals=self.core._molecule_info['num_orbitals'],
                         num_particles=self.core._molecule_info['num_particles'],
                         active_occupied=None, active_unoccupied=None,
                         initial_state=initial_state,
                         qubit_mapping=self.core._qubit_mapping,
                         two_qubit_reduction=self.core._two_qubit_reduction,
                         num_time_slices=1,
                         shallow_circuit_concat=False,
                         method_doubles='pucc',
                         excitation_type='d'
                         )

        algo = VQE(self.qubit_op, var_form, optimizer)
        result = algo.run(QuantumInstance(BasicAer.get_backend('statevector_simulator')))
        _, result = self.core.process_algorithm_result(result)
        self.assertAlmostEqual(result['energy'], self.reference_energy_pUCCD, places=6)

    def test_uccsd_hf_qUCCD0(self):
        """ singlet uccd test """

        optimizer = SLSQP(maxiter=100)
        initial_state = HartreeFock(self.qubit_op.num_qubits,
                                    self.core.molecule_info['num_orbitals'],
                                    self.core.molecule_info['num_particles'],
                                    qubit_mapping=self.core._qubit_mapping,
                                    two_qubit_reduction=self.core._two_qubit_reduction)

        var_form = UCCSD(num_qubits=self.qubit_op.num_qubits, depth=1,
                         num_orbitals=self.core._molecule_info['num_orbitals'],
                         num_particles=self.core._molecule_info['num_particles'],
                         active_occupied=None, active_unoccupied=None,
                         initial_state=initial_state,
                         qubit_mapping=self.core._qubit_mapping,
                         two_qubit_reduction=self.core._two_qubit_reduction,
                         num_time_slices=1,
                         shallow_circuit_concat=False,
                         method_doubles='succ',
                         excitation_type='d'
                         )

        algo = VQE(self.qubit_op, var_form, optimizer)
        result = algo.run(QuantumInstance(BasicAer.get_backend('statevector_simulator')))
        _, result = self.core.process_algorithm_result(result)
        self.assertAlmostEqual(result['energy'], self.reference_energy_UCCD0, places=6)

    def test_uccsd_hf_qUCCD0full(self):
        """ singlet full uccd test """

        optimizer = SLSQP(maxiter=100)

        initial_state = HartreeFock(self.qubit_op.num_qubits,
                                    self.core.molecule_info['num_orbitals'],
                                    self.core.molecule_info['num_particles'],
                                    qubit_mapping=self.core._qubit_mapping,
                                    two_qubit_reduction=self.core._two_qubit_reduction)

        var_form = UCCSD(num_qubits=self.qubit_op.num_qubits, depth=1,
                         num_orbitals=self.core._molecule_info['num_orbitals'],
                         num_particles=self.core._molecule_info['num_particles'],
                         active_occupied=None, active_unoccupied=None,
                         initial_state=initial_state,
                         qubit_mapping=self.core._qubit_mapping,
                         two_qubit_reduction=self.core._two_qubit_reduction,
                         num_time_slices=1,
                         shallow_circuit_concat=False,
                         method_doubles='succ_full',
                         excitation_type='d'
                         )

        algo = VQE(self.qubit_op, var_form, optimizer)
        result = algo.run(QuantumInstance(BasicAer.get_backend('statevector_simulator')))
        _, result = self.core.process_algorithm_result(result)
        self.assertAlmostEqual(result['energy'], self.reference_energy_UCCD0full, places=6)

    def test_uccsd_hf_qUCCSD(self):
        """ uccsd tapering test using all double excitations """

        # optimizer
        optimizer = SLSQP(maxiter=100)

        # initial state
        init_state = HartreeFock(num_qubits=self.the_tapered_op.num_qubits,
                                 num_orbitals=self.core._molecule_info['num_orbitals'],
                                 qubit_mapping=self.core._qubit_mapping,
                                 two_qubit_reduction=self.core._two_qubit_reduction,
                                 num_particles=self.core._molecule_info['num_particles'],
                                 sq_list=self.the_tapered_op.z2_symmetries.sq_list)

        var_form = UCCSD(num_qubits=self.the_tapered_op.num_qubits, depth=1,
                         num_orbitals=self.core._molecule_info['num_orbitals'],
                         num_particles=self.core._molecule_info['num_particles'],
                         active_occupied=None, active_unoccupied=None,
                         initial_state=init_state,
                         qubit_mapping=self.core._qubit_mapping,
                         two_qubit_reduction=self.core._two_qubit_reduction,
                         num_time_slices=1,
                         z2_symmetries=self.the_tapered_op.z2_symmetries,
                         shallow_circuit_concat=False,
                         method_doubles='ucc',
                         excitation_type='sd',
                         force_no_tap_excitation=True)

        algo = VQE(self.the_tapered_op, var_form, optimizer)

        result = algo.run(QuantumInstance(BasicAer.get_backend('statevector_simulator')))
        _, result = self.core.process_algorithm_result(result)
        self.assertAlmostEqual(result['energy'], self.reference_energy_UCCSD, places=6)

    def test_uccsd_hf_excitations(self):
        """ uccsd tapering test using all double excitations """

        # initial state
        init_state = HartreeFock(num_qubits=self.the_tapered_op.num_qubits,
                                 num_orbitals=self.core._molecule_info['num_orbitals'],
                                 qubit_mapping=self.core._qubit_mapping,
                                 two_qubit_reduction=self.core._two_qubit_reduction,
                                 num_particles=self.core._molecule_info['num_particles'],
                                 sq_list=self.the_tapered_op.z2_symmetries.sq_list)

        # check singlet excitations
        var_form = UCCSD(num_qubits=self.the_tapered_op.num_qubits, depth=1,
                         num_orbitals=self.core._molecule_info['num_orbitals'],
                         num_particles=self.core._molecule_info['num_particles'],
                         active_occupied=None, active_unoccupied=None,
                         initial_state=init_state,
                         qubit_mapping=self.core._qubit_mapping,
                         two_qubit_reduction=self.core._two_qubit_reduction,
                         num_time_slices=1,
                         z2_symmetries=self.the_tapered_op.z2_symmetries,
                         shallow_circuit_concat=False,
                         method_doubles='succ',
                         excitation_type='d',
                         force_no_tap_excitation=True)

        double_excitations_singlet = var_form._double_excitations
        res = TestUCCSDHartreeFock.excitation_lists_comparator(
            double_excitations_singlet, self.reference_singlet_double_excitations)
        self.assertEqual(res, True)

        # check grouped singlet excitations
        var_form = UCCSD(num_qubits=self.the_tapered_op.num_qubits, depth=1,
                         num_orbitals=self.core._molecule_info['num_orbitals'],
                         num_particles=self.core._molecule_info['num_particles'],
                         active_occupied=None, active_unoccupied=None,
                         initial_state=init_state,
                         qubit_mapping=self.core._qubit_mapping,
                         two_qubit_reduction=self.core._two_qubit_reduction,
                         num_time_slices=1,
                         z2_symmetries=self.the_tapered_op.z2_symmetries,
                         shallow_circuit_concat=False,
                         method_doubles='succ_full',
                         excitation_type='d',
                         force_no_tap_excitation=True)

        double_excitations_singlet_grouped = var_form._double_excitations_grouped
        res_groups = TestUCCSDHartreeFock.group_excitation_lists_comparator(
            double_excitations_singlet_grouped, self.reference_singlet_groups)
        self.assertEqual(res_groups, True)

    @staticmethod
    def pop_el_when_matched(list1, list2):
        """
        Compares if in list1 and list2 one of excitations is the same (regardless of permutations of
        its elements). When same excitation is found, it returns the 2 lists without that excitation
        .

        Args:
            list1 (list): list of excitations (e.g. [[0, 2, 4, 6], [0, 2, 4, 7]])
            list2 (list): list of excitations

        Returns:
            list: list1 with one popped element if match was found
            list: list2 with one popped element if match was found
        """
        counter = 0
        for i, exc1 in enumerate(list1):
            for j, exc2 in enumerate(list2):
                for ind1 in exc1:
                    for ind2 in exc2:
                        if ind1 == ind2:
                            counter += 1
                if counter == len(exc1) and counter == len(exc2):
                    list1.pop(i)
                    list2.pop(j)
                    break
            break
        return list1, list2

    @staticmethod
    def excitation_lists_comparator(list1, list2):
        """
        Compares if list1 and list2 contain same excitations (regardless of permutations of
        its elements). Only works provided all indices for an excitation are different.

        Args:
            list1 (list): list of excitations (e.g. [[0, 2, 4, 6], [0, 2, 4, 7]])
            list2 (list): list of excitations

        Returns:
            bool: True or False, if list1 and list2 contain the same excitations
        """
        if len(list1) != len(list2):
            return False

        number_el = len(list1)

        for _ in range(number_el):
            list1, list2 = TestUCCSDHartreeFock.pop_el_when_matched(list1, list2)

        return bool(len(list1) or len(list2) in [0])

    @staticmethod
    def group_excitation_lists_comparator(glist1, glist2):
        """
        Compares if list1 and list2 contain same excitations (regardless of permutations of
        its elements). Only works provided all indices for an excitation are different.

        Args:
            glist1 (list): list of excitations (e.g. [[0, 2, 4, 6], [0, 2, 4, 7]])
            glist2 (list): list of excitations

        Returns:
            bool: True or False, if list1 and list2 contain the same excitations
        """
        if len(glist1) != len(glist2):
            return False

        number_groups = len(glist1)
        counter = 0
        for _, gr1 in enumerate(glist1):
            for _, gr2 in enumerate(glist2):
                res = TestUCCSDHartreeFock.excitation_lists_comparator(gr1, gr2)
                if res is True:
                    counter += 1

        return bool(counter == number_groups)
