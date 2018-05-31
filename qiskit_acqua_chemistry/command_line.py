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

import argparse
import json
import logging
from qiskit_acqua_chemistry import ACQUAChemistry
from qiskit_acqua_chemistry._logging import build_logging_config,set_logger_config
from qiskit_acqua_chemistry.preferences import Preferences

def main():
    parser = argparse.ArgumentParser(description='Quantum Chemistry Program.')
    parser.add_argument('input', 
                        metavar='input', 
                        help='Chemistry Driver input or Algorithm JSON input file')
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('-o', 
                        metavar='output', 
                        help='Algorithm Results Output file name')
    group.add_argument('-jo', 
                        metavar='json output', 
                        help='Algorithm JSON Output file name')
    
    args = parser.parse_args()
    
    preferences = Preferences()
    if preferences.get_logging_config() is None:
        logging_config = build_logging_config(['qiskit_acqua_chemistry', 'qiskit_acqua'], logging.INFO)
        preferences.set_logging_config(logging_config)
        preferences.save()
    
    set_logger_config(preferences.get_logging_config())
    
    solver = ACQUAChemistry()
    
    # check to see if input is json file
    params = None
    try:
        with open(args.input) as json_file:
            params = json.load(json_file)
    except Exception as e:
        pass
    
    if params is not None:
        solver.run_algorithm_from_json(params, args.o)
    else:
        if args.jo is not None:
            solver.run_drive_to_jsonfile(args.input, args.jo)
        else:
            result = solver.run(args.input, args.o)
            if 'printable' in result:
                print('\n\n--------------------------------- R E S U L T ------------------------------------\n')
                for line in result['printable']:
                    print(line)

            

