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

"""Algorithm functions for running etc."""

import copy
import json
import logging

from qiskit.providers import BaseBackend
from qiskit.transpiler import PassManager

from .aqua_error import AquaError
from ._discover import (_discover_on_demand,
                        local_pluggables,
                        PluggableType,
                        get_pluggable_class)
from .utils.jsonutils import convert_dict_to_json, convert_json_to_dict
from .utils import CircuitCache
from .parser._inputparser import InputParser
from .parser import JSONSchema
from .quantum_instance import QuantumInstance
from .utils.backend_utils import (get_backend_from_provider,
                                  get_provider_from_backend,
                                  is_statevector_backend)

logger = logging.getLogger(__name__)


def run_algorithm(params, algo_input=None, json_output=False, backend=None):
    """
    Run algorithm as named in params.

    Using params and algo_input as input data and returning a result dictionary

    Args:
        params (dict): Dictionary of params for algo and dependent objects
        algo_input (AlgorithmInput): Main input data for algorithm. Optional, an algo may run entirely from params
        json_output (bool): False for regular python dictionary return, True for json conversion
        backend (BaseBackend or QuantumInstance): the experiemental settings to be used in place of backend name

    Returns:
        Result dictionary containing result of algorithm computation
    """
    qiskit_aqua = QiskitAqua(params, algo_input, backend)
    return qiskit_aqua.run(json_output)


def run_algorithm_to_json(params, algo_input=None, jsonfile='algorithm.json'):
    """
    Run algorithm as named in params.

    Using params and algo_input as input data
    and save the combined input as a json file. This json is self-contained and
    can later be used as a basis to call run_algorithm

    Args:
        params (dict): Dictionary of params for algo and dependent objects
        algo_input (AlgorithmInput): Main input data for algorithm. Optional, an algo may run entirely from params
        jsonfile (str): Name of file in which json should be saved

    Returns:
        Result dictionary containing the jsonfile name
    """
    return QiskitAqua.run_algorithm_to_json(params, algo_input, jsonfile)


class QiskitAqua(object):
    """Main Aqua class."""

    def __init__(self, params, algo_input=None, quantum_instance=None):
        """
        Create an QiskitAqua object

        Args:
            params (dict): Dictionary of params for algo and dependent objects
            algo_input (AlgorithmInput): Main input data for algorithm. Optional, an algo may run entirely from params
            quantum_instance (QuantumInstance or BaseBackend): the experiemental settings to be used in place of backend name
        """
        self._params = params
        self._algorithm_input = algo_input
        self._quantum_instance = None
        self._quantum_algorithm = None
        self._result = {}
        self._parser = None
        self._build_algorithm_from_dict(quantum_instance)

    @property
    def params(self):
        """Return Aqua params."""
        return self._params

    @property
    def algorithm_input(self):
        """Return Algorithm Input."""
        return self._algorithm_input

    @property
    def quantum_instance(self):
        """Return Quantum Instance."""
        return self._quantum_instance

    @property
    def quantum_algorithm(self):
        """Return Qusntum Algorithm."""
        return self._quantum_algorithm

    @property
    def result(self):
        """Return Experiment Result."""
        return self._result

    @property
    def json_result(self):
        """Return Experiment Result as JSON."""
        return convert_dict_to_json(self.result) if isinstance(self.result, dict) else None

    @property
    def parser(self):
        """Return Aqua parser."""
        return self._parser

    def _build_algorithm_from_dict(self, quantum_instance):
        _discover_on_demand()
        self._parser = InputParser(self._params)
        self._parser.parse()
        # before merging defaults attempts to find a provider for the backend in case no
        # provider was passed
        if quantum_instance is None and self._parser.get_section_property(JSONSchema.BACKEND, JSONSchema.PROVIDER) is None:
            backend_name = self._parser.get_section_property(JSONSchema.BACKEND, JSONSchema.NAME)
            if backend_name is not None:
                self._parser.set_section_property(JSONSchema.BACKEND, JSONSchema.PROVIDER,
                                                  get_provider_from_backend(backend_name))

        # check quantum_instance parameter
        backend = None
        if isinstance(quantum_instance, BaseBackend):
            backend = quantum_instance
        elif isinstance(quantum_instance, QuantumInstance):
            self._quantum_instance = quantum_instance
        elif quantum_instance is not None:
            raise AquaError('Invalid QuantumInstance or BaseBackend parameter {}.'.format(quantum_instance))

        # set provider and name in input file for proper backend schema dictionary build
        if backend is not None:
            self._parser.add_section_properties(JSONSchema.BACKEND,
                                                {
                                                    JSONSchema.PROVIDER: get_provider_from_backend(backend),
                                                    JSONSchema.NAME: backend.name(),
                                                })

        self._parser.validate_merge_defaults()
        logger.debug('Algorithm Input: {}'.format(json.dumps(self._parser.get_sections(), sort_keys=True, indent=4)))

        algo_name = self._parser.get_section_property(PluggableType.ALGORITHM.value, JSONSchema.NAME)
        if algo_name is None:
            raise AquaError('Missing algorithm name')

        if algo_name not in local_pluggables(PluggableType.ALGORITHM):
            raise AquaError('Algorithm "{0}" missing in local algorithms'.format(algo_name))

        if self._algorithm_input is None:
            input_name = self._parser.get_section_property('input', JSONSchema.NAME)
            if input_name is not None:
                input_params = copy.deepcopy(self._parser.get_section_properties('input'))
                del input_params[JSONSchema.NAME]
                convert_json_to_dict(input_params)
                self._algorithm_input = get_pluggable_class(PluggableType.INPUT, input_name).from_params(input_params)

        algo_params = copy.deepcopy(self._parser.get_sections())
        self._quantum_algorithm = get_pluggable_class(PluggableType.ALGORITHM,
                                                      algo_name).init_params(algo_params, self._algorithm_input)
        random_seed = self._parser.get_section_property(JSONSchema.PROBLEM, 'random_seed')
        self._quantum_algorithm.random_seed = random_seed

        if self._quantum_instance is not None:
            return

        # setup backend
        backend_provider = self._parser.get_section_property(JSONSchema.BACKEND, JSONSchema.PROVIDER)
        backend_name = self._parser.get_section_property(JSONSchema.BACKEND, JSONSchema.NAME)
        if backend_provider is not None and backend_name is not None:  # quantum algorithm
            if backend is None:
                backend = get_backend_from_provider(backend_provider, backend_name)

            backend_cfg = {k: v for k, v in self._parser.get_section(JSONSchema.BACKEND).items() if
                           k not in [JSONSchema.PROVIDER, JSONSchema.NAME]}

            # set shots for state vector
            if is_statevector_backend(backend):
                backend_cfg['shots'] = 1

            # check coupling map
            if 'coupling_map_from_device' in backend_cfg:
                coupling_map_from_device = backend_cfg.get('coupling_map_from_device')
                del backend_cfg['coupling_map_from_device']
                if coupling_map_from_device is not None:
                    names = coupling_map_from_device.split(':')
                    if len(names) == 2:
                        device_backend = get_backend_from_provider(names[0], names[1])
                        device_coupling_map = device_backend.configuration().coupling_map
                        if device_coupling_map is not None:
                            coupling_map = backend_cfg.get('coupling_map')
                            if coupling_map is None:
                                backend_cfg['coupling_map'] = device_coupling_map
                            else:
                                if coupling_map != device_coupling_map:
                                    logger.warning("Coupling map '{}' used instead of device coupling map '{}'.".format(coupling_map, device_coupling_map))

            # check noise model
            if 'noise_model' in backend_cfg:
                noise_model = backend_cfg.get('noise_model')
                del backend_cfg['noise_model']
                if noise_model is not None:
                    names = noise_model.split(':')
                    if len(names) == 2:
                        # Generate an Aer noise model for device
                        from qiskit.providers.aer import noise
                        device_backend = get_backend_from_provider(names[0], names[1])
                        noise_model = noise.device.basic_device_noise_model(device_backend.properties())
                        noise_basis_gates = None
                        if noise_model is not None and noise_model.basis_gates is not None:
                            noise_basis_gates = noise_model.basis_gates
                            noise_basis_gates = noise_basis_gates.split(',') if isinstance(noise_basis_gates, str) else noise_basis_gates
                        if noise_basis_gates is not None:
                            basis_gates = backend_cfg.get('basis_gates')
                            if basis_gates is None:
                                backend_cfg['basis_gates'] = noise_basis_gates
                            else:
                                if basis_gates != noise_basis_gates:
                                    logger.warning("Basis gates '{}' used instead of noise model basis gates '{}'.".format(basis_gates, noise_basis_gates))

            backend_cfg['seed_mapper'] = random_seed
            pass_manager = PassManager() if backend_cfg.pop('skip_transpiler', False) else None
            if pass_manager is not None:
                backend_cfg['pass_manager'] = pass_manager

            backend_cfg['backend'] = backend
            backend_cfg['seed'] = random_seed
            backend_cfg['skip_qobj_validation'] = self._parser.get_section_property(JSONSchema.PROBLEM, 'skip_qobj_validation')
            use_caching = self._parser.get_section_property(JSONSchema.PROBLEM, 'circuit_caching')
            if use_caching:
                deepcopy_qobj = self._parser.get_section_property(JSONSchema.PROBLEM, 'skip_qobj_deepcopy')
                cache_file = self._parser.get_section_property(JSONSchema.PROBLEM, 'circuit_cache_file')
                backend_cfg['circuit_cache'] = CircuitCache(skip_qobj_deepcopy=deepcopy_qobj, cache_file=cache_file)

            self._quantum_instance = QuantumInstance(**backend_cfg)

    def run(self, json_output=False):
        if self.quantum_algorithm is None:
            raise AquaError('Missing Quantum Algorithm.')

        self._result = self.quantum_algorithm.run(self.quantum_instance)
        return self.json_result if json_output else self.result

    @staticmethod
    def run_algorithm_to_json(params, algo_input=None, jsonfile='algorithm.json'):
        """
        Run algorithm as named in params.

        Using params and algo_input as input data
        and save the combined input as a json file. This json is self-contained and
        can later be used as a basis to call run_algorithm

        Args:
            params (dict): Dictionary of params for algo and dependent objects
            algo_input (AlgorithmInput): Main input data for algorithm. Optional, an algo may run entirely from params
            jsonfile (str): Name of file in which json should be saved

        Returns:
            Result dictionary containing the jsonfile name
        """
        _discover_on_demand()

        inputparser = InputParser(params)
        inputparser.parse()
        inputparser.validate_merge_defaults()

        algo_params = copy.deepcopy(inputparser.get_sections())

        if algo_input is not None:
            input_params = algo_input.to_params()
            convert_dict_to_json(input_params)
            algo_params['input'] = input_params
            algo_params['input']['name'] = algo_input.configuration['name']

        logger.debug('Result: {}'.format(json.dumps(algo_params, sort_keys=True, indent=4)))
        with open(jsonfile, 'w') as fp:
            json.dump(algo_params, fp, sort_keys=True, indent=4)

        logger.info("Algorithm input file saved: '{}'".format(jsonfile))

        return {'jsonfile': jsonfile}
