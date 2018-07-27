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

"""
InputParser test.
"""

import unittest
import os
import json

from test.common import QiskitAquaTestCase
from qiskit_aqua import AlgorithmError
from qiskit_aqua import run_algorithm
from qiskit_aqua.parser import InputParser

class TestInputParser(QiskitAquaTestCase):
    """Input Parser and algorithms tests."""

    def setUp(self):
        filepath = self._get_resource_path('H2-0.735.json')
        self.parser = InputParser(filepath)
        self.parser.parse()

    def test_save(self):
        save_path = self._get_resource_path('output.txt')
        self.parser.save_to_file(save_path)

        p = InputParser(save_path)
        p.parse()
        os.remove(save_path)
        dict1 = json.loads(json.dumps(self.parser.get_sections()))
        dict2 = json.loads(json.dumps(p.get_sections()))
        self.assertEqual(dict1,dict2)

    def test_load_from_dict(self):
        json_dict = self.parser.get_sections()

        p = InputParser(json_dict)
        p.parse()
        dict1 = json.loads(json.dumps(self.parser.get_sections()))
        dict2 = json.loads(json.dumps(p.get_sections()))
        self.assertEqual(dict1,dict2)

    def test_is_modified(self):
        json_dict = self.parser.get_sections()

        p = InputParser(json_dict)
        p.parse()
        p.set_section_property('optimizer','maxfun',1002)
        self.assertTrue(p.is_modified())
        self.assertEqual(p.get_section_property('optimizer','maxfun'),1002)

    def test_validate(self):
        json_dict = self.parser.get_sections()

        p = InputParser(json_dict)
        p.parse()
        try:
            p.validate_merge_defaults()
        except Exception as e:
            self.fail(str(e))

        p.set_section_property('optimizer','dummy',1002)
        self.assertRaises(AlgorithmError, p.validate_merge_defaults)

    def test_run_algorithm(self):
        filepath = self._get_resource_path('ExactEigensolver.json')
        params = None
        with open(filepath) as json_file:
            params = json.load(json_file)

        dict_ret = None
        try:
            dict_ret = run_algorithm(params, None, False)
        except Exception as e:
            self.fail(str(e))

        self.assertIsInstance(dict_ret, dict)

if __name__ == '__main__':
    unittest.main()
