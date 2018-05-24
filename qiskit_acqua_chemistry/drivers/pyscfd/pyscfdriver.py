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

from qiskit_acqua_chemistry.drivers import BaseDriver
from qiskit_acqua_chemistry.drivers.pyscfd.integrals import compute_integrals


class PySCFDriver(BaseDriver):
    """Python implementation of a PySCF driver."""

    def __init__(self, configuration=None):
        """
        Args:
            configuration (dict): driver configuration
        """
        super(PySCFDriver, self).__init__(configuration)

    def run(self, section):
        return compute_integrals(section['properties'])


