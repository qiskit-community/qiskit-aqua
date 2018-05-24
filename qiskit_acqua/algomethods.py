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

from qiskit_acqua.algorithmerror import AlgorithmError
from qiskit_acqua._discover import (_discover_on_demand,
                                  local_algorithms,
                                  get_algorithm_instance)
from qiskit_acqua.utils.jsonutils import convert_dict_to_json
from qiskit_acqua.parser._inputparser import InputParser
from qiskit_acqua.input import get_input_instance
import logging
import json
import copy

logger = logging.getLogger(__name__)


def run_algorithm(params, algo_input = None,json_output=False):
    _discover_on_demand()

    inputparser = InputParser(params)
    inputparser.parse()
    inputparser.validate_merge_defaults()
    logger.debug('Algorithm Input: {}'.format(json.dumps(inputparser.get_sections(), sort_keys=True, indent=4)))

    algo_name = inputparser.get_section_property(InputParser.ALGORITHM, InputParser.NAME)
    if algo_name is None:
        raise AlgorithmError('Missing algorithm name')

    if algo_name not in local_algorithms():
        raise AlgorithmError('Algorithm "{0}" missing in local algorithms'.format(algo_name))

    backend = inputparser.get_section_property(InputParser.BACKEND, InputParser.NAME)
    if backend is None:
        raise AlgorithmError('Missing backend name')

    backend_cfg = {k: v for k, v in inputparser.get_section(InputParser.BACKEND).items() if k != 'name'}
    backend_cfg['backend'] = backend

    algorithm = get_algorithm_instance(algo_name)
    algorithm.random_seed = inputparser.get_section_property(InputParser.PROBLEM, 'random_seed')
    algorithm.setup_quantum_backend(**backend_cfg)

    algo_params = copy.deepcopy(inputparser.get_sections())

    if algo_input is None:
        input_name = inputparser.get_section_property('input', InputParser.NAME)
        if input_name is not None:
            algo_input = get_input_instance(input_name)
            input_params = copy.deepcopy(inputparser.get_section_properties('input'))
            del input_params[InputParser.NAME]
            algo_input.from_params(input_params)

    algorithm.init_params(algo_params, algo_input)
    value = algorithm.run()
    if isinstance(value, dict) and json_output:
        convert_dict_to_json(value)

    return value
