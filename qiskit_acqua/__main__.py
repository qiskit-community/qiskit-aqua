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

import sys
import os
import argparse

algo_directory = os.path.dirname(os.path.realpath(__file__))
algo_directory = os.path.join(algo_directory,'..')
sys.path.insert(0,algo_directory)

parser = argparse.ArgumentParser(description='QISKit Acqua Program.')
parser.add_argument('input', 
                    metavar='input', 
                    help='Algorithm JSON input file')
parser.add_argument('-jo', 
                    metavar='output', 
                    help='Algorithm JSON output file name',
                    required=False)

args = parser.parse_args()

import json
import logging
from qiskit_acqua._logging import build_logging_config,set_logger_config
from qiskit_acqua.preferences import Preferences
from qiskit_acqua import run_algorithm
from qiskit_acqua.utils import convert_json_to_dict

preferences = Preferences()
if preferences.get_logging_config() is None:
    logging_config = build_logging_config(['qiskit_acqua'],logging.INFO)
    preferences.set_logging_config(logging_config)
    preferences.save()

set_logger_config(preferences.get_logging_config())

params = None
with open(args.input) as json_file:
    params = json.load(json_file)

ret = run_algorithm(params,None,True)

if args.jo is not None:
    with open(args.jo, 'w') as f:
        print('{}'.format(ret), file=f)
else:
    convert_json_to_dict(ret)
    print('Output:')
    if isinstance(ret,dict):
        for k,v in ret.items():
            print("'{}': {}".format(k,v))
    else:
        print(ret)
            

