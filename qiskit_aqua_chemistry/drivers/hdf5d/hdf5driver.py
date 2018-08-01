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

from qiskit_aqua_chemistry.drivers import BaseDriver
import logging
from qiskit_aqua_chemistry import QMolecule
from qiskit_aqua_chemistry import AquaChemistryError
import os

logger = logging.getLogger(__name__)

class HDF5Driver(BaseDriver):
    """Python implementation of a hdf5 driver."""
    
    KEY_HDF5_INPUT = 'hdf5_input'

    def __init__(self, configuration=None):
        """
        Args:
            configuration (dict): driver configuration
        """
        super(HDF5Driver, self).__init__(configuration)

    def run(self, section):
        properties = section['properties']
        if HDF5Driver.KEY_HDF5_INPUT not in properties:
            raise AquaChemistryError('Missing hdf5 input property')
       
        hdf5_file = properties[HDF5Driver.KEY_HDF5_INPUT]
        if self.work_path is not None and not os.path.isabs(hdf5_file):
            hdf5_file = os.path.abspath(os.path.join(self.work_path,hdf5_file))
         
        if not os.path.isfile(hdf5_file):
            raise LookupError('HDF5 file not found: {}'.format(hdf5_file))
                            
        molecule = QMolecule(hdf5_file)
        molecule.load()
        return molecule
