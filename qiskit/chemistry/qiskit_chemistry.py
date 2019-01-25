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

from .qiskit_chemistry_error import QiskitChemistryError
from qiskit.chemistry.drivers import local_drivers, get_driver_class
from qiskit_aqua import run_algorithm, get_provider_from_backend
from qiskit_aqua.utils import convert_json_to_dict
from qiskit.chemistry.parser import InputParser
from qiskit_aqua.parser import JSONSchema
import json
import os
import copy
import pprint
import logging
from qiskit.chemistry.core import get_chemistry_operator_class

logger = logging.getLogger(__name__)


class QiskitChemistry(object):
    """Main entry point."""

    KEY_HDF5_OUTPUT = 'hdf5_output'
    _DRIVER_RUN_TO_HDF5 = 1
    _DRIVER_RUN_TO_ALGO_INPUT = 2

    def __init__(self):
        """Create an QiskitChemistry object."""
        self._parser = None
        self._core = None

    def run(self, input, output=None, backend=None):
        """
        Runs the Aqua Chemistry experiment

        Args:
            input (dictionary/filename): Input data
            output (filename):  Output data
            backend (BaseBackend): backend object

        Returns:
            result dictionary
        """
        if input is None:
            raise QiskitChemistryError("Missing input.")

        self._parser = InputParser(input)
        self._parser.parse()
        driver_return = self._run_driver_from_parser(self._parser, False)
        if driver_return[0] == QiskitChemistry._DRIVER_RUN_TO_HDF5:
            logger.info('No further process.')
            return {'printable': [driver_return[1]]}

        data = run_algorithm(driver_return[1], driver_return[2], True, backend)
        if not isinstance(data, dict):
            raise QiskitChemistryError("Algorithm run result should be a dictionary")

        convert_json_to_dict(data)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('Algorithm returned: {}'.format(pprint.pformat(data, indent=4)))

        lines, result = self._format_result(data)
        logger.info('Processing complete. Final result available')
        result['printable'] = lines

        if output is not None:
            with open(output, 'w') as f:
                for line in lines:
                    print(line, file=f)

        return result

    def save_input(self, input_file):
        """
        Save the input of a run to a file.

        Params:
            input_file (string): file path
        """
        if self._parser is None:
            raise QiskitChemistryError("Missing input information.")

        self._parser.save_to_file(input_file)

    def run_drive_to_jsonfile(self, input, jsonfile):
        if jsonfile is None:
            raise QiskitChemistryError("Missing json file")

        data = self._run_drive(input, True)
        if data is None:
            logger.info('No data to save. No further process.')
            return

        with open(jsonfile, 'w') as fp:
            json.dump(data, fp, sort_keys=True, indent=4)

        print("Algorithm input file saved: '{}'".format(jsonfile))

    def run_algorithm_from_jsonfile(self, jsonfile, output=None, backend=None):
        """
        Runs the Aqua Chemistry experiment from json file

        Args:
            jsonfile (filename): Input data
            output (filename):  Output data
            backend (BaseBackend): backend object

        Returns:
            result dictionary
        """
        with open(jsonfile) as json_file:
            return self.run_algorithm_from_json(json.load(json_file), output, backend)

    def run_algorithm_from_json(self, params, output=None, backend=None):
        """
        Runs the Aqua Chemistry experiment from json dictionary

        Args:
            params (dictionary): Input data
            output (filename):  Output data
            backend (BaseBackend): backend object

        Returns:
            result dictionary
        """
        ret = run_algorithm(params, None, True, backend)
        if not isinstance(ret, dict):
            raise QiskitChemistryError("Algorithm run result should be a dictionary")

        convert_json_to_dict(ret)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('Algorithm returned: {}'.format(pprint.pformat(ret, indent=4)))

        print('Output:')
        if isinstance(ret, dict):
            for k, v in ret.items():
                print("'{}': {}".format(k, v))
        else:
            print(ret)

        return ret

    def _format_result(self, data):
        lines, result = self._core.process_algorithm_result(data)
        return lines, result

    def run_drive(self, input):
        return self._run_drive(input, False)

    def _run_drive(self, input, save_json_algo_file):
        if input is None:
            raise QiskitChemistryError("Missing input.")

        self._parser = InputParser(input)
        self._parser.parse()
        driver_return = self._run_driver_from_parser(self._parser, save_json_algo_file)
        driver_return[1]['input'] = driver_return[2].to_params()
        driver_return[1]['input']['name'] = driver_return[2].configuration['name']
        return driver_return[1]

    def _run_driver_from_parser(self, p, save_json_algo_file):
        if p is None:
            raise QiskitChemistryError("Missing parser")

        # before merging defaults attempts to find a provider for the backend in case no
        # provider was passed
        if p.get_section_property(JSONSchema.BACKEND, JSONSchema.PROVIDER) is None:
            backend_name = p.get_section_property(JSONSchema.BACKEND, JSONSchema.NAME)
            if backend_name is not None:
                p.set_section_property(JSONSchema.BACKEND, JSONSchema.PROVIDER, get_provider_from_backend(backend_name))

        p.validate_merge_defaults()
        # logger.debug('ALgorithm Input Schema: {}'.format(json.dumps(p..get_sections(), sort_keys=True, indent=4)))

        experiment_name = "-- no &NAME section found --"
        if JSONSchema.NAME in p.get_section_names():
            name_sect = p.get_section(JSONSchema.NAME)
            if name_sect is not None:
                experiment_name = str(name_sect)
        logger.info('Running chemistry problem from input file: {}'.format(p.get_filename()))
        logger.info('Experiment description: {}'.format(experiment_name.rstrip()))

        driver_name = p.get_section_property(InputParser.DRIVER, JSONSchema.NAME)
        if driver_name is None:
            raise QiskitChemistryError('Property "{0}" missing in section "{1}"'.format(JSONSchema.NAME, InputParser.DRIVER))

        hdf5_file = p.get_section_property(InputParser.DRIVER, QiskitChemistry.KEY_HDF5_OUTPUT)

        if driver_name not in local_drivers():
            raise QiskitChemistryError('Driver "{0}" missing in local drivers'.format(driver_name))

        work_path = None
        input_file = p.get_filename()
        if input_file is not None:
            work_path = os.path.dirname(os.path.realpath(input_file))

        section = p.get_section(driver_name)
        driver = get_driver_class(driver_name).init_from_input(section)
        driver.work_path = work_path
        molecule = driver.run()

        if work_path is not None and hdf5_file is not None and not os.path.isabs(hdf5_file):
            hdf5_file = os.path.abspath(os.path.join(work_path, hdf5_file))

        molecule.log()

        if hdf5_file is not None:
            molecule._origin_driver_name = driver_name
            molecule._origin_driver_config = section if isinstance(section, str) else json.dumps(section, sort_keys=True, indent=4)
            molecule.save(hdf5_file)
            text = "HDF5 file saved '{}'".format(hdf5_file)
            logger.info(text)
            if not save_json_algo_file:
                logger.info('Run ended with hdf5 file saved.')
                return QiskitChemistry._DRIVER_RUN_TO_HDF5, text

        # Run the Hamiltonian to process the QMolecule and get an input for algorithms
        cls = get_chemistry_operator_class(p.get_section_property(InputParser.OPERATOR, JSONSchema.NAME))
        self._core = cls.init_params(p.get_section_properties(InputParser.OPERATOR))
        input_object = self._core.run(molecule)

        logger.debug('Core computed substitution variables {}'.format(self._core.molecule_info))
        result = p.process_substitutions(self._core.molecule_info)
        logger.debug('Substitutions {}'.format(result))

        params = {}
        for section_name, section in p.get_sections().items():
            if section_name == JSONSchema.NAME or \
               section_name == InputParser.DRIVER or \
               section_name == driver_name.lower() or \
               section_name == InputParser.OPERATOR or \
               not isinstance(section, dict):
                continue

            params[section_name] = copy.deepcopy(section)
            if JSONSchema.PROBLEM == section_name and InputParser.AUTO_SUBSTITUTIONS in params[section_name]:
                del params[section_name][InputParser.AUTO_SUBSTITUTIONS]

        return QiskitChemistry._DRIVER_RUN_TO_ALGO_INPUT, params, input_object
