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

""" Qiskit Cmemistry Main Entry points """

import json
import os
import copy
import pprint
import logging
from qiskit.providers import BaseBackend
from qiskit.chemistry.drivers import local_drivers, get_driver_class
from qiskit.aqua import QiskitAqua, get_provider_from_backend
from qiskit.aqua.input import EnergyInput
from qiskit.chemistry.parser import InputParser
from qiskit.aqua.parser import JSONSchema
from qiskit.chemistry.core import get_chemistry_operator_class
from .qiskit_chemistry_error import QiskitChemistryError

logger = logging.getLogger(__name__)


def run_experiment(params, output=None, backend=None):
    """
    Run Chemistry from params.

    Using params and returning a result dictionary

    Args:
        params (Union(dictionary, filename)): Chemistry input data
        output (filename):  Output data
        backend (QuantumInstance or BaseBackend): the experimental
                        settings to be used in place of backend name

    Returns:
        Dict: Result dictionary containing result of chemistry computation
    """
    qiskit_chemistry = QiskitChemistry()
    return qiskit_chemistry.run(params, output, backend)


def run_driver_to_json(params, jsonfile='algorithm.json'):
    """
    Runs the Aqua Chemistry driver only

    Args:
        params (Union(dictionary/filename)): Chemistry input data
        jsonfile (filename):  Name of file that will contain the Aqua JSON input data

    Returns:
        Dict: Result dictionary containing the jsonfile name
    """
    qiskit_chemistry = QiskitChemistry()
    qiskit_chemistry.run_driver(params)
    data = copy.deepcopy(qiskit_chemistry.qiskit_aqua.params)
    data['input'] = qiskit_chemistry.qiskit_aqua.algorithm_input.to_params()
    data['input']['name'] = qiskit_chemistry.qiskit_aqua.algorithm_input.configuration['name']
    with open(jsonfile, 'w') as file:
        json.dump(data, file, sort_keys=True, indent=4)

    print("Algorithm input file saved: '{}'".format(jsonfile))
    return {'jsonfile': jsonfile}


class QiskitChemistry:
    """Main Chemistry class."""

    def __init__(self):
        """Create an QiskitChemistry object."""
        self._parser = None
        self._operator = None
        self._qiskit_aqua = None
        self._hdf5_file = None
        self._chemistry_result = None

    @property
    def qiskit_aqua(self):
        """Returns Qiskit Aqua object."""
        return self._qiskit_aqua

    @property
    def hdf5_file(self):
        """Returns Chemistry hdf5 path with chemistry results, if used."""
        return self._hdf5_file

    @property
    def operator(self):
        """Returns Chemistry Operator."""
        return self._operator

    @property
    def chemistry_result(self):
        """Returns Chemistry result."""
        return self._chemistry_result

    @property
    def parser(self):
        """Returns Chemistry parser."""
        return self._parser

    def run(self, params, output=None, backend=None):
        """
        Runs the Qiskit Chemistry experiment

        Args:
            params (Union(dictionary, filename)): Chemistry input data
            output (filename):  Output data
            backend (QuantumInstance or BaseBackend): the experimental settings
                to be used in place of backend name

        Returns:
            Dict: result dictionary
        Raises:
            QiskitChemistryError: Missing Input, QiskitAqua object not created,
                result should be ditctionary
        """
        if params is None:
            raise QiskitChemistryError("Missing input.")

        self.run_driver(params, backend)
        if self.hdf5_file:
            logger.info('No further process.')
            self._chemistry_result = {'printable': ["HDF5 file saved '{}'".format(self.hdf5_file)]}
            return self.chemistry_result

        if self.qiskit_aqua is None:
            raise QiskitChemistryError("QiskitAqua object not created.")

        data = self.qiskit_aqua.run()
        if not isinstance(data, dict):
            raise QiskitChemistryError("Algorithm run result should be a dictionary")

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('Algorithm returned: %s', pprint.pformat(data, indent=4))

        lines, self._chemistry_result = self.operator.process_algorithm_result(data)
        logger.info('Processing complete. Final result available')
        self._chemistry_result['printable'] = lines

        if output is not None:
            with open(output, 'w') as file:
                for line in self.chemistry_result['printable']:
                    print(line, file=file)

        return self.chemistry_result

    def run_driver(self, params, backend=None):
        """
        Runs the Qiskit Chemistry driver

        Args:
            params (Union(dictionary, filename)): Chemistry input data
            backend (QuantumInstance or BaseBackend): the experimental settings
                to be used in place of backend name
         Raises:
            QiskitChemistryError: Missing Input
        """
        if params is None:
            raise QiskitChemistryError("Missing input.")

        self._operator = None
        self._chemistry_result = None
        self._qiskit_aqua = None
        self._hdf5_file = None
        self._parser = InputParser(params)
        self._parser.parse()

        # before merging defaults attempts to find a provider for the backend in case no
        # provider was passed
        if backend is None and \
                self._parser.get_section_property(JSONSchema.BACKEND, JSONSchema.PROVIDER) is None:
            backend_name = self._parser.get_section_property(JSONSchema.BACKEND, JSONSchema.NAME)
            if backend_name is not None:
                self._parser.set_section_property(JSONSchema.BACKEND, JSONSchema.PROVIDER,
                                                  get_provider_from_backend(backend_name))

        # set provider and name in input file for proper backend schema dictionary build
        if isinstance(backend, BaseBackend):
            self._parser.backend = backend
            self._parser.add_section_properties(JSONSchema.BACKEND,
                                                {
                                                    JSONSchema.PROVIDER:
                                                        get_provider_from_backend(backend),
                                                    JSONSchema.NAME: backend.name(),
                                                })

        self._parser.validate_merge_defaults()

        experiment_name = "-- no &NAME section found --"
        if JSONSchema.NAME in self._parser.get_section_names():
            name_sect = self._parser.get_section(JSONSchema.NAME)
            if name_sect is not None:
                experiment_name = str(name_sect)
        logger.info('Running chemistry problem from input file: %s', self._parser.get_filename())
        logger.info('Experiment description: %s', experiment_name.rstrip())

        driver_name = self._parser.get_section_property(InputParser.DRIVER, JSONSchema.NAME)
        if driver_name is None:
            raise QiskitChemistryError('Property "{0}" missing in section "{1}"'.format(
                JSONSchema.NAME, InputParser.DRIVER))

        self._hdf5_file = \
            self._parser.get_section_property(InputParser.DRIVER, InputParser.HDF5_OUTPUT)

        if driver_name not in local_drivers():
            raise QiskitChemistryError('Driver "{0}" missing in local drivers'.format(driver_name))

        work_path = None
        input_file = self._parser.get_filename()
        if input_file is not None:
            work_path = os.path.dirname(os.path.realpath(input_file))

        section = self._parser.get_section(driver_name)
        driver = get_driver_class(driver_name).init_from_input(section)
        driver.work_path = work_path
        molecule = driver.run()

        if work_path is not None and \
                self._hdf5_file is not None and not os.path.isabs(self._hdf5_file):
            self._hdf5_file = os.path.abspath(os.path.join(work_path, self._hdf5_file))

        molecule.log()

        if self._hdf5_file is not None:
            molecule.save(self._hdf5_file)
            logger.info("HDF5 file saved '%s'", self._hdf5_file)

        # Run the Hamiltonian to process the QMolecule and get an input for algorithms
        clazz = get_chemistry_operator_class(
            self._parser.get_section_property(InputParser.OPERATOR, JSONSchema.NAME))
        self._operator = clazz.init_params(
            self._parser.get_section_properties(InputParser.OPERATOR))
        qubit_op, aux_ops = self.operator.run(molecule)
        input_object = EnergyInput(qubit_op, aux_ops)

        logger.debug('Core computed substitution variables %s', self.operator.molecule_info)
        result = self._parser.process_substitutions(self.operator.molecule_info)
        logger.debug('Substitutions %s', result)

        aqua_params = {}
        for section_name, section in self._parser.get_sections().items():
            if section_name == JSONSchema.NAME or \
               section_name == InputParser.DRIVER or \
               section_name == driver_name.lower() or \
               section_name == InputParser.OPERATOR or \
               not isinstance(section, dict):
                continue

            aqua_params[section_name] = copy.deepcopy(section)
            if JSONSchema.PROBLEM == section_name and \
                    InputParser.AUTO_SUBSTITUTIONS in aqua_params[section_name]:
                del aqua_params[section_name][InputParser.AUTO_SUBSTITUTIONS]

        self._qiskit_aqua = QiskitAqua(aqua_params, input_object, backend)
