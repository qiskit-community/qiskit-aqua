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

from qiskit.chemistry.components.initial_states import VSCF


class TestInitialStateVSCF(QiskitChemistryTestCase):
    """ Initial State vscf tests """

    def test_qubits_4(self):
        """ 2 modes 2 modals - test """
        basis = [2, 2]
        vscf = VSCF(basis)
        cct = vscf.construct_circuit('vector')
        np.testing.assert_array_equal(cct, [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def test_qubits_5(self):
        """ 2 modes 2 modals for the first mode and 3 modals for the second - test """
        basis = [2, 3]
        vscf = VSCF(basis)
        cct = vscf.construct_circuit('vector')
        np.testing.assert_array_equal(cct, [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


if __name__ == '__main__':
    unittest.main()
