# -*- coding: utf-8 -*-

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

"""
Code inside the test is the chemistry sample from the readme.
If this test fails and code changes are needed here to resolve
the issue then ensure changes are made to readme too.
"""

import unittest

from test.chemistry import QiskitChemistryTestCase
from qiskit.chemistry import QiskitChemistryError


class TestReadmeSample(QiskitChemistryTestCase):
    """Test sample code from readme"""

    def setUp(self):
        super().setUp()
        try:
            # pylint: disable=import-outside-toplevel
            from qiskit.chemistry.drivers import PySCFDriver
            PySCFDriver(atom='Li .0 .0 .0; H .0 .0 1.6')
        except QiskitChemistryError:
            self.skipTest('PYSCF driver does not appear to be installed')

        try:
            # pylint: disable=import-outside-toplevel
            # pylint: disable=unused-import
            from qiskit import Aer
        except Exception as ex:  # pylint: disable=broad-except
            self.skipTest("Aer doesn't appear to be installed. Error: '{}'".format(str(ex)))
            return

    def test_readme_sample(self):
        """ readme sample test """
        # pylint: disable=import-outside-toplevel,redefined-builtin

        def print(*args):
            """ overloads print to log values """
            if args:
                self.log.debug(args[0], *args[1:])

        # --- Exact copy of sample code ----------------------------------------

        from qiskit.chemistry import FermionicOperator
        from qiskit.chemistry.drivers import PySCFDriver, UnitsType
        from qiskit.aqua.operators import Z2Symmetries

        # Use PySCF, a classical computational chemistry software
        # package, to compute the one-body and two-body integrals in
        # molecular-orbital basis, necessary to form the Fermionic operator
        driver = PySCFDriver(atom='H .0 .0 .0; H .0 .0 0.735',
                             unit=UnitsType.ANGSTROM,
                             basis='sto3g')
        molecule = driver.run()
        num_particles = molecule.num_alpha + molecule.num_beta
        num_spin_orbitals = molecule.num_orbitals * 2

        # Build the qubit operator, which is the input to the VQE algorithm in Aqua
        ferm_op = FermionicOperator(h1=molecule.one_body_integrals, h2=molecule.two_body_integrals)
        map_type = 'PARITY'
        qubit_op = ferm_op.mapping(map_type)
        qubit_op = Z2Symmetries.two_qubit_reduction(qubit_op, num_particles)
        num_qubits = qubit_op.num_qubits

        # setup a classical optimizer for VQE
        from qiskit.aqua.components.optimizers import L_BFGS_B
        optimizer = L_BFGS_B()

        # setup the initial state for the variational form
        from qiskit.chemistry.components.initial_states import HartreeFock
        init_state = HartreeFock(num_qubits, num_spin_orbitals, num_particles)

        # setup the variational form for VQE
        from qiskit.aqua.components.variational_forms import RYRZ
        var_form = RYRZ(num_qubits, initial_state=init_state)

        # setup and run VQE
        from qiskit.aqua.algorithms import VQE
        algorithm = VQE(qubit_op, var_form, optimizer)

        # set the backend for the quantum computation
        from qiskit import Aer
        backend = Aer.get_backend('statevector_simulator')

        result = algorithm.run(backend)
        print(result.eigenvalue.real)

        # ----------------------------------------------------------------------

        self.assertAlmostEqual(result.eigenvalue.real, -1.8572750301938803, places=6)


if __name__ == '__main__':
    unittest.main()
