# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test Initial State HartreeFock """

import unittest
from test.chemistry import QiskitChemistryTestCase
import numpy as np
from ddt import ddt, idata, unpack
from qiskit.chemistry.components.initial_states import HartreeFock
from qiskit.aqua.operators import op_converter
from qiskit.chemistry import QiskitChemistryError
from qiskit.chemistry.drivers import PySCFDriver, UnitsType
from qiskit.chemistry.core import Hamiltonian, TransformationType, QubitMappingType


@ddt
class TestInitialStateHartreeFock(QiskitChemistryTestCase):
    """ Initial State HartreeFock tests """

    def test_qubits_4_jw_h2(self):
        """ qubits 4 jw h2 test """
        hrfo = HartreeFock(4, 4, [1, 1], 'jordan_wigner', False)
        cct = hrfo.construct_circuit('vector')
        np.testing.assert_array_equal(cct, [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def test_qubits_4_py_h2(self):
        """ qubits 4 py h2 test """
        hrfo = HartreeFock(4, 4, [1, 1], 'parity', False)
        cct = hrfo.construct_circuit('vector')
        np.testing.assert_array_equal(cct, [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def test_qubits_4_bk_h2(self):
        """ qubits 4 bk h2 test """
        hrfo = HartreeFock(4, 4, [1, 1], 'bravyi_kitaev', False)
        cct = hrfo.construct_circuit('vector')
        np.testing.assert_array_equal(cct, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def test_qubits_2_py_h2(self):
        """ qubits 2 py h2 test """
        hrfo = HartreeFock(2, 4, 2, 'parity', True)
        cct = hrfo.construct_circuit('vector')
        np.testing.assert_array_equal(cct, [0.0, 1.0, 0.0, 0.0])

    def test_qubits_2_py_h2_cct(self):
        """ qubits 2 py h2 cct test """
        hrfo = HartreeFock(2, 4, [1, 1], 'parity', True)
        cct = hrfo.construct_circuit('circuit')
        self.assertEqual(cct.qasm(), 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\n'
                                     'u3(3.141592653589793,0.0,3.141592653589793) q[0];\n')

    def test_qubits_6_py_lih_cct(self):
        """ qubits 6 py lih cct test """
        hrfo = HartreeFock(6, 10, [1, 1], 'parity', True, [1, 2])
        cct = hrfo.construct_circuit('circuit')
        self.assertEqual(cct.qasm(), 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[6];\n'
                                     'u3(3.141592653589793,0.0,3.141592653589793) q[0];\n'
                                     'u3(3.141592653589793,0.0,3.141592653589793) q[1];\n')

    def test_qubits_10_bk_lih_bitstr(self):
        """ qubits 10 bk lih bitstr test """
        hrfo = HartreeFock(10, 10, [1, 1], 'bravyi_kitaev', False)
        bitstr = hrfo.bitstr
        np.testing.assert_array_equal(bitstr,
                                      [False, False, False, False, True,
                                       False, True, False, True, True])

    @idata([
        [QubitMappingType.JORDAN_WIGNER],
        [QubitMappingType.PARITY],
        [QubitMappingType.BRAVYI_KITAEV]
    ])
    @unpack
    def test_hf_value(self, mapping):
        """ hf value test """
        try:
            driver = PySCFDriver(atom='Li .0 .0 .0; H .0 .0 1.6',
                                 unit=UnitsType.ANGSTROM,
                                 charge=0,
                                 spin=0,
                                 basis='sto3g')
        except QiskitChemistryError:
            self.skipTest('PYSCF driver does not appear to be installed')
        qmolecule = driver.run()
        core = Hamiltonian(transformation=TransformationType.FULL,
                           qubit_mapping=mapping,
                           two_qubit_reduction=False,
                           freeze_core=False,
                           orbital_reduction=[])

        qubit_op, _ = core.run(qmolecule)
        qubit_op = op_converter.to_matrix_operator(qubit_op)
        hrfo = HartreeFock(qubit_op.num_qubits, core.molecule_info['num_orbitals'],
                           core.molecule_info['num_particles'], mapping.value, False)
        qc = hrfo.construct_circuit('vector')
        hf_energy = qubit_op.evaluate_with_statevector(qc)[0].real + core._nuclear_repulsion_energy

        self.assertAlmostEqual(qmolecule.hf_energy, hf_energy, places=8)


if __name__ == '__main__':
    unittest.main()
