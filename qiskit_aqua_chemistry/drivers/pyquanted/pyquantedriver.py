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
from qiskit_aqua_chemistry import AquaChemistryError
from qiskit_aqua_chemistry.drivers.pyquanted.integrals import compute_integrals
import importlib
import logging

logger = logging.getLogger(__name__)


class PyQuanteDriver(BaseDriver):
    """Python implementation of a PyQuante driver."""

    CONFIGURATION = {
        "name": "PYQUANTE",
        "description": "PyQuante Driver",
        "input_schema": {
            "$schema": "http://json-schema.org/schema#",
            "id": "pyquante_schema",
            "type": "object",
            "properties": {
                "atoms": {
                    "type": "string",
                    "default": "H 0.0 0.0 0.0; H 0.0 0.0 0.735"
                },
                "units": {
                    "type": "string",
                    "default": "Angstrom",
                    "oneOf": [
                         {"enum": ["Angstrom", "Bohr"]}
                    ]
                },
                "charge": {
                    "type": "integer",
                    "default": 0
                },
                "multiplicity": {
                    "type": "integer",
                    "default": 1
                },
                "basis": {
                    "type": "string",
                    "default": "sto3g",
                    "oneOf": [
                         {"enum": ["sto3g", "6-31g", "6-31g**"]}
                    ]
                }
            },
            "additionalProperties": False
        }
    }

    def __init__(self):
        super().__init__()

    @staticmethod
    def check_driver_valid():
        err_msg = 'PyQuante2 is not installed. See https://github.com/rpmuller/pyquante2'
        try:
            spec = importlib.util.find_spec('pyquante2')
            if spec is not None:
                return
        except Exception as e:
            logger.debug('PyQuante2 check error {}'.format(str(e)))
            raise AquaChemistryError(err_msg) from e

        raise AquaChemistryError(err_msg)

    def run(self, section):
        return compute_integrals(section['properties'])
