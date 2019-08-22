# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test Grouped Paulis """

import unittest
from test.aqua.common import QiskitAquaTestCase
from qiskit.quantum_info import pauli_group
from qiskit.aqua import Operator


class TestGroupedPaulis(QiskitAquaTestCase):
    """Grouped Pauli tests."""

    def test_grouped_paulis(self):
        """ grouped paulis test """
        n = 3  # number of qubits

        p_g = pauli_group(n, case='tensor')
        self.assertTrue(p_g != -1, "Error in pauli_group()")

        p_g = [[1.0, x] for x in p_g]  # create paulis with equal weight
        self.log.debug("Number of Paulis: %s", len(p_g))

        ham_op_original = Operator(paulis=p_g, coloring=None)
        ham_op_original._paulis_to_grouped_paulis()
        gp_original = ham_op_original.grouped_paulis

        ham_op_new = Operator(paulis=p_g, coloring="largest-degree")
        ham_op_new._paulis_to_grouped_paulis()
        gp_new = ham_op_new.grouped_paulis

        self.log.debug("#groups in original= %s #groups in new=%s",
                       len(gp_original), len(gp_new))

        self.log.debug("------- Original --------")
        for each in gp_original:
            for x in each:
                self.log.debug('%s %s', x[0], x[1].to_label())
            self.log.debug('---')

        self.log.debug("-------- New ------------")
        for each in gp_new:
            for x in each:
                self.log.debug('%s %s', x[0], x[1].to_label())
            self.log.debug('---')


if __name__ == '__main__':
    unittest.main()
