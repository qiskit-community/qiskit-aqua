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
import tkinter as tk

_ROOT = None


def main():
    global _ROOT
    _ROOT = tk.Tk()
    _ROOT.withdraw()
    _ROOT.update_idletasks()
    _ROOT.after(0, main_chemistry)
    _ROOT.mainloop()


def main_chemistry():
    try:
        from qiskit_chemistry import QiskitChemistry
        from qiskit_chemistry._logging import get_logging_level, build_logging_config, set_logging_config
        from qiskit_chemistry.preferences import Preferences
        parser = argparse.ArgumentParser(description='Qiskit Chemistry Command Line Tool')
        parser.add_argument('input',
                            metavar='input',
                            help='Chemistry input file or saved JSON input file')
        group = parser.add_mutually_exclusive_group(required=False)
        group.add_argument('-o',
                           metavar='output',
                           help='Algorithm Results Output file name')
        group.add_argument('-jo',
                           metavar='json output',
                           help='Algorithm JSON Output file name')

        args = parser.parse_args()

        # update logging setting with latest external packages
        preferences = Preferences()
        logging_level = logging.INFO
        if preferences.get_logging_config() is not None:
            set_logging_config(preferences.get_logging_config())
            logging_level = get_logging_level()

        preferences.set_logging_config(build_logging_config(logging_level))
        preferences.save()

        set_logging_config(preferences.get_logging_config())

        solver = QiskitChemistry()

        # check to see if input is json file
        params = None
        try:
            with open(args.input) as json_file:
                params = json.load(json_file)
        except:
            pass

        if params is not None:
            solver.run_algorithm_from_json(params, args.o)
        else:
            if args.jo is not None:
                solver.run_drive_to_jsonfile(args.input, args.jo)
            else:
                result = solver.run(args.input, args.o)
                if result is not None and 'printable' in result:
                    print('\n\n--------------------------------- R E S U L T ------------------------------------\n')
                    for line in result['printable']:
                        print(line)
    finally:
        global _ROOT
        if _ROOT is not None:
            _ROOT.destroy()
            _ROOT = None
