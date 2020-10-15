# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test GroundStateEigensolver """

import unittest

from test.chemistry import QiskitChemistryTestCase

from qiskit import BasicAer
from qiskit.aqua import QuantumInstance
from qiskit.chemistry import QiskitChemistryError
from qiskit.chemistry.drivers import PySCFDriver, UnitsType
from qiskit.chemistry.transformations import FermionicTransformation
from qiskit.chemistry.transformations.fermionic_transformation import QubitMappingType
from qiskit.chemistry.algorithms.ground_state_solvers import GroundStateEigensolver
from qiskit.chemistry.algorithms.ground_state_solvers.minimum_eigensolver_factories import \
    (VQEUCCSDFactory, NumPyMinimumEigensolverFactory)


class TestGroundStateEigensolver(QiskitChemistryTestCase):
    """ Test GroundStateEigensolver """

    def setUp(self):
        super().setUp()
        try:
            self.driver = PySCFDriver(atom='H .0 .0 .0; H .0 .0 0.735',
                                      unit=UnitsType.ANGSTROM,
                                      charge=0,
                                      spin=0,
                                      basis='sto3g')
        except QiskitChemistryError:
            self.skipTest('PYSCF driver does not appear to be installed')

        self.reference_energy = -1.137306

        self.transformation = FermionicTransformation(qubit_mapping=QubitMappingType.JORDAN_WIGNER)

    def test_npme(self):
        """ Test NumPyMinimumEigensolver """
        solver = NumPyMinimumEigensolverFactory()
        calc = GroundStateEigensolver(self.transformation, solver)
        res = calc.solve(self.driver)
        self.assertAlmostEqual(res.total_energies[0], self.reference_energy, places=6)

    def test_npme_with_default_filter(self):
        """ Test NumPyMinimumEigensolver with default filter """
        solver = NumPyMinimumEigensolverFactory(use_default_filter_criterion=True)
        calc = GroundStateEigensolver(self.transformation, solver)
        res = calc.solve(self.driver)
        self.assertAlmostEqual(res.total_energies[0], self.reference_energy, places=6)

    def test_vqe_uccsd(self):
        """ Test VQE UCCSD case """
        solver = VQEUCCSDFactory(QuantumInstance(BasicAer.get_backend('statevector_simulator')))
        calc = GroundStateEigensolver(self.transformation, solver)
        res = calc.solve(self.driver)
        self.assertAlmostEqual(res.total_energies[0], self.reference_energy, places=6)

    def _setup_evaluation_operators(self):
        # first we run a ground state calculation
        solver = VQEUCCSDFactory(QuantumInstance(BasicAer.get_backend('statevector_simulator')))
        calc = GroundStateEigensolver(self.transformation, solver)
        res = calc.solve(self.driver)

        # now we decide that we want to evaluate another operator
        # for testing simplicity, we just use some pre-constructed auxiliary operators
        _, aux_ops = self.transformation.transform(self.driver)
        return calc, res, aux_ops

    def test_eval_op_single(self):
        """ Test evaluating a single additional operator """
        calc, res, aux_ops = self._setup_evaluation_operators()
        # we filter the list because in this test we test a single operator evaluation
        add_aux_op = aux_ops[0][0]

        # now we have the ground state calculation evaluate it
        add_aux_op_res = calc.evaluate_operators(res.raw_result.eigenstate, add_aux_op)
        self.assertIsInstance(add_aux_op_res[0], complex)
        self.assertAlmostEqual(add_aux_op_res[0].real, 2, places=6)

    def test_eval_op_list(self):
        """ Test evaluating a list of additional operator """
        calc, res, aux_ops = self._setup_evaluation_operators()
        # we filter the list because of simplicity
        expected_results = {'number of particles': 2,
                            's^2': 0,
                            'magnetization': 0}
        add_aux_op = aux_ops[0:3]

        # now we have the ground state calculation evaluate them
        add_aux_op_res = calc.evaluate_operators(res.raw_result.eigenstate, add_aux_op)
        self.assertIsInstance(add_aux_op_res, list)
        # in this list we require that the order of the results remains unchanged
        for idx, expected in enumerate(expected_results.values()):
            self.assertAlmostEqual(add_aux_op_res[idx][0].real, expected, places=6)

    def test_eval_op_dict(self):
        """ Test evaluating a dict of additional operator """
        calc, res, aux_ops = self._setup_evaluation_operators()
        # we filter the list because of simplicity
        expected_results = {'number of particles': 2,
                            's^2': 0,
                            'magnetization': 0}
        add_aux_op = aux_ops[0:3]
        # now we convert it into a dictionary
        add_aux_op = dict(zip(expected_results.keys(), add_aux_op))

        # now we have the ground state calculation evaluate them
        add_aux_op_res = calc.evaluate_operators(res.raw_result.eigenstate, add_aux_op)
        self.assertIsInstance(add_aux_op_res, dict)
        for name, expected in expected_results.items():
            self.assertAlmostEqual(add_aux_op_res[name][0].real, expected, places=6)


if __name__ == '__main__':
    unittest.main()
