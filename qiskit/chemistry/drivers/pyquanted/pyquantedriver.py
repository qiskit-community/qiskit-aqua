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

from qiskit.chemistry.drivers import BaseDriver, UnitsType
from qiskit.chemistry import QiskitChemistryError
from qiskit.chemistry.drivers.pyquanted.integrals import compute_integrals
import importlib
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class BasisType(Enum):
    BSTO3G = 'sto3g'
    B631G = '6-31g'
    B631GSS = '6-31g**'


class PyQuanteDriver(BaseDriver):
    """Python implementation of a PyQuante driver."""

    KEY_UNITS = 'units'
    KEY_BASIS = 'basis'

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
                KEY_UNITS: {
                    "type": "string",
                    "default": UnitsType.ANGSTROM.value,
                    "oneOf": [
                         {"enum": [
                            UnitsType.ANGSTROM.value,
                            UnitsType.BOHR.value,
                         ]}
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
                KEY_BASIS: {
                    "type": "string",
                    "default": BasisType.BSTO3G.value,
                    "oneOf": [
                         {"enum": [
                             BasisType.BSTO3G.value,
                             BasisType.B631G.value,
                             BasisType.B631GSS.value,
                         ]}
                    ]
                }
            },
            "additionalProperties": False
        }
    }

    def __init__(self,
                 atoms,
                 units=UnitsType.ANGSTROM,
                 charge=0,
                 multiplicity=1,
                 basis=BasisType.BSTO3G):
        """
        Initializer
        Args:
            atoms (str or list): atoms list or string separated by semicolons or line breaks
            units (UnitsType): angstrom or bohr
            charge (int): charge
            multiplicity (int): spin multiplicity
            basis (BasisType): sto3g or 6-31g or 6-31g**
        """
        if not isinstance(atoms, list) and not isinstance(atoms, str):
            raise QiskitChemistryError("Invalid atom input for PYQUANTE Driver '{}'".format(atoms))

        if isinstance(atoms, list):
            atoms = ';'.join(atoms)
        else:
            atoms = atoms.replace('\n', ';')

        units = units.value
        basis = basis.value

        self.validate(locals())
        super().__init__()
        self._atoms = atoms
        self._units = units
        self._charge = charge
        self._multiplicity = multiplicity
        self._basis = basis

    @staticmethod
    def check_driver_valid():
        err_msg = 'PyQuante2 is not installed. See https://github.com/rpmuller/pyquante2'
        try:
            spec = importlib.util.find_spec('pyquante2')
            if spec is not None:
                return
        except Exception as e:
            logger.debug('PyQuante2 check error {}'.format(str(e)))
            raise QiskitChemistryError(err_msg) from e

        raise QiskitChemistryError(err_msg)

    @classmethod
    def init_from_input(cls, section):
        """
        Initialize via section dictionary.

        Args:
            params (dict): section dictionary

        Returns:
            Driver: Driver object
        """
        if section is None or not isinstance(section, dict):
            raise QiskitChemistryError('Invalid or missing section {}'.format(section))

        params = section
        kwargs = {}
        for k, v in params.items():
            if k == PyQuanteDriver.KEY_UNITS:
                v = UnitsType(v)
            elif k == PyQuanteDriver.KEY_BASIS:
                v = BasisType(v)

            kwargs[k] = v

        logger.debug('init_from_input: {}'.format(kwargs))
        return cls(**kwargs)

    def run(self):
        return compute_integrals(atoms=self._atoms,
                                 units=self._units,
                                 charge=self._charge,
                                 multiplicity=self._multiplicity,
                                 basis=self._basis)
