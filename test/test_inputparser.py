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
from qiskit_acqua_chemistry.parser import InputParser

class TestInputParser(QISKitAcquaChemistryTestCase):
    """InputParser tests."""

    def test_parse(self):
        filepath = self._get_resource_path('input.txt')
        parser = InputParser(filepath)
        parser.parse()
        for name in parser.get_section_names():
            self.log.debug(parser.get_section(name))
            
if __name__ == '__main__':
    unittest.main()
