# -*- coding: utf-8 -*-

# Copyright 2018 IBM.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import unittest

from qiskit.tools.qi.pauli import pauli_group

from test.common import QiskitAquaTestCase
from qiskit_aqua.operator import Operator


class TestGroupedPaulis(QiskitAquaTestCase):
    """GroupedPaulki tests."""

    def test_grouped_paulis(self):
        n = 3  # number of qubits

        pg = pauli_group(n, case=1)
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
