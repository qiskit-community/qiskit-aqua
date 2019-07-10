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

import unittest

from qiskit.quantum_info import pauli_group

from test.common import QiskitAquaTestCase
from qiskit.aqua import Operator


class TestGroupedPaulis(QiskitAquaTestCase):
    """GroupedPaulki tests."""

    def test_grouped_paulis(self):
        n = 3  # number of qubits

        pg = pauli_group(n, case='tensor')
        self.assertTrue(pg != -1, "Error in pauli_group()")

        pg = [[1.0, x] for x in pg]  # create paulis with equal weight
        self.log.debug("Number of Paulis: {}".format(len(pg)))

        hamOpOriginal = Operator(paulis=pg, coloring=None)
        hamOpOriginal._paulis_to_grouped_paulis()
        gpOriginal = hamOpOriginal.grouped_paulis

        hamOpNew = Operator(paulis=pg, coloring="largest-degree")
        hamOpNew._paulis_to_grouped_paulis()
        gpNew = hamOpNew.grouped_paulis

        self.log.debug("#groups in original= {} #groups in new={}".format(len(gpOriginal), len(gpNew)))

        self.log.debug("------- Original --------")
        for each in gpOriginal:
            for x in each:
                self.log.debug('{} {}'.format(x[0], x[1].to_label()))
            self.log.debug('---')

        self.log.debug("-------- New ------------")
        for each in gpNew:
            for x in each:
                self.log.debug('{} {}'.format(x[0], x[1].to_label()))
            self.log.debug('---')


if __name__ == '__main__':
    unittest.main()
