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

"""
InputParser test.
"""

import unittest
from test.common import QiskitChemistryTestCase
from qiskit.aqua import AquaError
from qiskit.chemistry.parser import InputParser
import os
import json


class TestInputParser(QiskitChemistryTestCase):
    """InputParser tests."""

    def setUp(self):
        super().setUp()
        filepath = self._get_resource_path('test_input_parser.txt')
        self.parser = InputParser(filepath)
        self.parser.parse()

    def test_save(self):
        save_path = self._get_resource_path('output.txt')
        self.parser.save_to_file(save_path)

        p = InputParser(save_path)
        p.parse()
        os.remove(save_path)
        dict1 = json.loads(json.dumps(self.parser.to_dictionary()))
        dict2 = json.loads(json.dumps(p.to_dictionary()))
        self.assertEqual(dict1, dict2)

    def test_load_from_dict(self):
        json_dict = self.parser.get_sections()

        p = InputParser(json_dict)
        p.parse()
        dict1 = json.loads(json.dumps(self.parser.to_dictionary()))
        dict2 = json.loads(json.dumps(p.to_dictionary()))
        self.assertEqual(dict1, dict2)

    def test_is_modified(self):
        json_dict = self.parser.get_sections()

        p = InputParser(json_dict)
        p.parse()
        p.set_section_property('optimizer', 'maxfun', 1002)
        self.assertTrue(p.is_modified())
        self.assertEqual(p.get_section_property('optimizer', 'maxfun'), 1002)

    def test_validate(self):
        json_dict = self.parser.get_sections()

        p = InputParser(json_dict)
        p.parse()
        try:
            p.validate_merge_defaults()
        except Exception as e:
            self.fail(str(e))

        with self.assertRaises(AquaError):
            p.set_section_property('backend', 'max_credits', -1)


if __name__ == '__main__':
    unittest.main()
