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

from .qiskit_aqua import QiskitAqua


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
