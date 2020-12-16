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

""" Test Analytic QEOM VQE """
import unittest
from test.chemistry import QiskitChemistryTestCase

from qiskit.chemistry.algorithms.excited_states_solvers import ExcitedStatesEigensolver
from qiskit.chemistry.algorithms.excited_states_solvers import AnalyticQEOM
from qiskit.chemistry.algorithms.ground_state_solvers import (GroundStateEigensolver,
                                                              VQEUCCSDFactory)
from qiskit.aqua import QuantumInstance
from qiskit import BasicAer
from qiskit.chemistry.algorithms import NumPyEigensolverFactory
from qiskit.chemistry.drivers import PySCFDriver, UnitsType, Molecule
from qiskit.chemistry.transformations import FermionicTransformation, FermionicQubitMappingType


class TestAnalyticQEOMVQE(QiskitChemistryTestCase):
    """ Test Analytic QEOM VQE """

    def test_analytic_qeom_vqe(self):
        """ test Analytic QEOM VQE """

        geo = [['C', (0.0000, 0.0000, 0.5000)],
               ['H', (0.0000, 0.9289, 1.0626)],
               ['H', (0.0000, -0.9289, 1.0626)],
               ['C', (0.0000, 0.0000, -0.5000)],
               ['H', (0.0000, 0.9289, -1.0626)],
               ['H', (0.0000, -0.9289, -1.0626)]]

        molecule = Molecule(geometry=geo, charge=0, multiplicity=1)
        driver = PySCFDriver(molecule=molecule, unit=UnitsType.ANGSTROM, basis='sto-6g')
        transformation = FermionicTransformation(
            qubit_mapping=FermionicQubitMappingType.PARITY,
            two_qubit_reduction=True,
            orbital_reduction=[x for x in range(14) if x not in [7, 8]])

        numpy_solver = NumPyEigensolverFactory(use_default_filter_criterion=True)

        # This first part sets the ground state solver
        # see more about this part in the ground state calculation tutorial
        quantum_instance = QuantumInstance(BasicAer.get_backend('statevector_simulator'))
        solver = VQEUCCSDFactory(quantum_instance)
        gsc = GroundStateEigensolver(transformation, solver)

        # The qEOM algorithm is simply instantiated with the chosen ground state solver
        qeom_excited_states_calculation = AnalyticQEOM(gsc, 'sd')

        numpy_excited_states = ExcitedStatesEigensolver(transformation, numpy_solver)
        numpy_results = numpy_excited_states.solve(driver)

        qeom_results = qeom_excited_states_calculation.solve(driver)

        print(numpy_results)
        print("\n************************************************\n")
        print(qeom_results)


if __name__ == '__main__':
    unittest.main()
