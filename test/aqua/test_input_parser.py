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

""" Test InputParser """

import unittest
import os
import json
from test.aqua.common import QiskitAquaTestCase
from qiskit.aqua import AquaError
from qiskit.aqua.parser._inputparser import InputParser
from qiskit.aqua.algorithms import ExactEigensolver
from qiskit.aqua.operators import WeightedPauliOperator


class TestInputParser(QiskitAquaTestCase):
    """Input Parser and algorithms tests."""

    def setUp(self):
        super().setUp()
        filepath = self._get_resource_path('H2-0.735.json')
        self.parser = InputParser(filepath)
        self.parser.parse()

    def test_save(self):
        """ save test """
        save_path = self._get_resource_path('output.txt')
        self.parser.save_to_file(save_path)

        parse = InputParser(save_path)
        parse.parse()
        os.remove(save_path)
        dict1 = json.loads(json.dumps(self.parser.get_sections()))
        dict2 = json.loads(json.dumps(parse.get_sections()))
        self.assertEqual(dict1, dict2)

    def test_load_from_dict(self):
        """ load from dict test """
        json_dict = self.parser.get_sections()

        parse = InputParser(json_dict)
        parse.parse()
        dict1 = json.loads(json.dumps(self.parser.get_sections()))
        dict2 = json.loads(json.dumps(parse.get_sections()))
        self.assertEqual(dict1, dict2)

    def test_is_modified(self):
        """ is modified test """
        json_dict = self.parser.get_sections()

        parse = InputParser(json_dict)
        parse.parse()
        parse.set_section_property('optimizer', 'maxfun', 1002)
        self.assertTrue(parse.is_modified())
        self.assertEqual(parse.get_section_property('optimizer', 'maxfun'), 1002)

    def test_validate(self):
        """ validate test """
        json_dict = self.parser.get_sections()

        parse = InputParser(json_dict)
        parse.parse()
        try:
            parse.validate_merge_defaults()
        except Exception as ex:  # pylint: disable=broad-except
            self.fail(str(ex))

        with self.assertRaises(AquaError):
            parse.set_section_property('backend', 'max_credits', -1)

    def test_run_algorithm(self):
        """ run algorithm test """
        pauli_dict_zz = {
            'paulis': [
                {"coeff": {"imag": 0.0, "real": 1.0}, "label": "ZZ"}
            ]
        }
        qubit_op_zz = WeightedPauliOperator.from_dict(pauli_dict_zz)
        exact_eigensolver = ExactEigensolver(qubit_op_zz, k=1)
        dict_ret = exact_eigensolver.run()
        self.assertIsInstance(dict_ret, dict)


if __name__ == '__main__':
    unittest.main()
