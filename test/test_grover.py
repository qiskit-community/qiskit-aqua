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
import operator

from parameterized import parameterized
from test.common import QiskitAquaTestCase
from qiskit_aqua import get_algorithm_instance, get_oracle_instance


class TestGrover(QiskitAquaTestCase):

    @parameterized.expand([
        ['test_grover_tiny.cnf', 1],
        ['test_grover.cnf', 2]
    ])
    def test_grover(self, input_file, num_iterations=1):
        self.log.debug('Testing Grover search with {} iteration(s) on SAT problem instance: \n{}'.format(
            num_iterations, open(input_file).read(),
        ))
        # get ground-truth
        with open(input_file) as f:
            header = f.readline()
            self.assertGreaterEqual(header.find('solution'), 0, 'Ground-truth info missing.')
        self.groundtruth = [
            ''.join([
                '1' if i > 0 else '0'
                for i in sorted([int(v) for v in s.strip().split() if v != '0'], key=abs)
            ])[::-1]
            for s in header.split('solutions:' if header.find('solutions:') >= 0 else 'solution:')[-1].split(',')
        ]
        sat_oracle = get_oracle_instance('SAT')
        with open(input_file) as f:
            sat_oracle.init_args(f.read())

        grover = get_algorithm_instance('Grover')
        grover.setup_quantum_backend(backend='local_qasm_simulator', shots=100)
        grover.init_args(sat_oracle, num_iterations=num_iterations)

        ret = grover.run()

        self.log.debug('Ground-truth Solutions: {}.'.format(self.groundtruth))
        self.log.debug('Measurement result:     {}.'.format(ret['measurements']))
        top_measurement = max(ret['measurements'].items(), key=operator.itemgetter(1))[0]
        self.log.debug('Top measurement:        {}.'.format(top_measurement))
        self.log.debug('Search Result:          {}.'.format(ret['result']))
        self.assertIn(top_measurement, self.groundtruth)


if __name__ == '__main__':
    unittest.main()
