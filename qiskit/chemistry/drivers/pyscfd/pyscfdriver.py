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

from qiskit.chemistry.drivers import BaseDriver, UnitsType
from qiskit.chemistry import QiskitChemistryError
from qiskit.chemistry.drivers.pyscfd.integrals import compute_integrals
import importlib
import logging

logger = logging.getLogger(__name__)


class PySCFDriver(BaseDriver):
    """Python implementation of a PySCF driver."""

    CONFIGURATION = {
        "name": "PYSCF",
        "description": "PYSCF Driver",
        "input_schema": {
            "$schema": "http://json-schema.org/schema#",
            "id": "pyscf_schema",
            "type": "object",
            "properties": {
                "atom": {
                    "type": "string",
                    "default": "H 0.0 0.0 0.0; H 0.0 0.0 0.735"
                },
                "unit": {
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
                "spin": {
                    "type": "integer",
                    "default": 0
                },
                "basis": {
                    "type": "string",
                    "default": "sto3g"
                },
                "max_memory": {
                    "type": ["integer", "null"],
                    "default": None
                }
            },
            "additionalProperties": False
        }
    }

    def __init__(self,
                 atom,
                 unit=UnitsType.ANGSTROM,
                 charge=0,
                 spin=0,
                 basis='sto3g',
                 max_memory=None):
        """
        Initializer
        Args:
            atom (str or list): atom list or string separated by semicolons or line breaks
            unit (UnitsType): angstrom or bohr
            charge (int): charge
            spin (int): spin
            basis (str): basis set
            max_memory (int): maximum memory
        """
        if not isinstance(atom, list) and not isinstance(atom, str):
            raise QiskitChemistryError("Invalid atom input for PYSCF Driver '{}'".format(atom))

        if isinstance(atom, list):
            atom = ';'.join(atom)
        else:
            atom = atom.replace('\n', ';')

        unit = unit.value
        self.validate(locals())
        super().__init__()
        self._atom = atom
        self._unit = unit
        self._charge = charge
        self._spin = spin
        self._basis = basis
        self._max_memory = max_memory

    @staticmethod
    def check_driver_valid():
        err_msg = "PySCF is not installed. Use 'pip install pyscf'"
        try:
            spec = importlib.util.find_spec('pyscf')
            if spec is not None:
                return
        except Exception as e:
            logger.debug('PySCF check error {}'.format(str(e)))
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
            if k == 'unit':
                v = UnitsType(v)

            kwargs[k] = v

        logger.debug('init_from_input: {}'.format(kwargs))
        return cls(**kwargs)

    def run(self):
        return compute_integrals(atom=self._atom,
                                 unit=self._unit,
                                 charge=self._charge,
                                 spin=self._spin,
                                 basis=self._basis,
                                 max_memory=self._max_memory)
