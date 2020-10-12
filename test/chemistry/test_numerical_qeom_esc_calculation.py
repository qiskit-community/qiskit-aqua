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

""" Test NumericalqEOM excited states calculation """

import unittest
from test.chemistry import QiskitChemistryTestCase
from qiskit.aqua.algorithms import NumPyMinimumEigensolver
from qiskit.chemistry import QiskitChemistryError
from qiskit.chemistry.drivers import PySCFDriver, UnitsType
from qiskit.chemistry.qubit_transformations import FermionicTransformation
from qiskit.chemistry.qubit_transformations.fermionic_transformation import QubitMappingType
from qiskit.chemistry.ground_state_calculation import MinimumEigensolverGroundStateCalculation
from qiskit.chemistry.excited_states_calculation import NumericalqEOMExcitedStatesCalculation

class TestNumericalqEOMESCCalculation(QiskitChemistryTestCase):
    """ Test NumericalqEOM excited states calculation """

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

    def test_numpy_mes(self):

        solver = NumPyMinimumEigensolver()
        gsc = MinimumEigensolverGroundStateCalculation(self.transformation, solver)
        esc = NumericalqEOMExcitedStatesCalculation(gsc, 'sd')
        results = esc.compute_excitedstates(self.driver)

        print(results)

if __name__ == '__main__':
    unittest.main()