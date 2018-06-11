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
from test.common import QISKitAcquaChemistryTestCase
from qiskit_acqua_chemistry import ACQUAChemistryError
from qiskit_acqua_chemistry.parser import InputParser
import os
import json

class TestInputParser(QISKitAcquaChemistryTestCase):
    """InputParser tests."""
    
    def setUp(self):
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
        self.assertEqual(dict1,dict2)
        
    def test_load_from_dict(self):
        json_dict = self.parser.to_JSON()
        
        p = InputParser(json_dict)
        p.parse()
        dict1 = json.loads(json.dumps(self.parser.to_dictionary()))
        dict2 = json.loads(json.dumps(p.to_dictionary()))
        self.assertEqual(dict1,dict2)
        
    def test_is_modified(self):
        json_dict = self.parser.to_JSON()
        
        p = InputParser(json_dict)
        p.parse()
        p.set_section_property('optimizer','maxfun',1002)
        self.assertTrue(p.is_modified())
        self.assertEqual(p.get_section_property('optimizer','maxfun'),1002)
        
    def test_validate(self):
        json_dict = self.parser.to_JSON()
        
        p = InputParser(json_dict)
        p.parse()
        try:
            p.validate_merge_defaults()
        except Exception as e:
            self.fail(str(e))
            
        p.set_section_property('optimizer','dummy',1002)
        self.assertRaises(ACQUAChemistryError, p.validate_merge_defaults)
            
if __name__ == '__main__':
    unittest.main()
