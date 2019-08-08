# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import unittest
from test.aqua.common import QiskitAquaTestCase
from qiskit.aqua import (local_pluggables_types,
                         local_pluggables,
                         get_pluggable_class,
                         get_pluggable_configuration,
                         PluggableType)
from qiskit.aqua.input import AlgorithmInput
import inspect
import os
import subprocess

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


class TestConfigurationIntegrity(QiskitAquaTestCase):

    @staticmethod
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, env=env,
                                cwd=os.path.join(ROOT_DIR))
        stdout, stderr = proc.communicate()
        if proc.returncode > 0:
            raise OSError('Command {} exited with code {}: {}'.format(
                cmd, proc.returncode, stderr.strip().decode('ascii')))
        return stdout

    # TODO: disable this test for now, run locally because devs may have a hard time finding the cause
    def disable_test_load_aqua(self):
        try:
            out = TestConfigurationIntegrity._minimal_ext_cmd(['python', os.path.join(ROOT_DIR, 'load_aqua.py')])
            out = out.strip().decode('ascii')
            if out:
                self.fail('One or more qiskit imports are interfering with Aqua loading: {}'.format(out))
        except OSError as ex:
            self.fail(str(ex))

    def test_pluggable_inputs(self):
        algorithm_problems = set()
        for pluggable_name in local_pluggables(PluggableType.ALGORITHM):
            configuration = get_pluggable_configuration(PluggableType.ALGORITHM, pluggable_name)
            if isinstance(configuration, dict):
                algorithm_problems.update(configuration.get('problems', []))

        err_msgs = []
        all_problems = set()
        for pluggable_name in local_pluggables(PluggableType.INPUT):
            cls = get_pluggable_class(PluggableType.INPUT, pluggable_name)
            configuration = get_pluggable_configuration(PluggableType.INPUT, pluggable_name)
            missing_problems = []
            if isinstance(configuration, dict):
                problem_names = configuration.get('problems', [])
                all_problems.update(problem_names)
                for problem_name in problem_names:
                    if problem_name not in algorithm_problems:
                        missing_problems.append(problem_name)

            if len(missing_problems) > 0:
                err_msgs.append("{}: No algorithm declares the problems {}.".format(cls, missing_problems))

        invalid_problems = list(set(AlgorithmInput._PROBLEM_SET).difference(all_problems))
        if len(invalid_problems) > 0:
            err_msgs.append("Base Class AlgorithmInput contains problems {} that don't belong to any Input class.".format(invalid_problems))

        if len(err_msgs) > 0:
            self.fail('\n'.join(err_msgs))

    def test_pluggable_configuration(self):
        err_msgs = []
        for pluggable_type in local_pluggables_types():
            for pluggable_name in local_pluggables(pluggable_type):
                cls = get_pluggable_class(pluggable_type, pluggable_name)
                configuration = get_pluggable_configuration(pluggable_type, pluggable_name)
                if not isinstance(configuration, dict):
                    err_msgs.append("{} configuration isn't a dictionary.".format(cls))
                    continue

                if pluggable_type in [PluggableType.ALGORITHM, PluggableType.INPUT]:
                    if len(configuration.get('problems', [])) == 0:
                        err_msgs.append("{} missing or empty 'problems' section.".format(cls))

                schema_found = False
                for configuration_name, configuration_value in configuration.items():
                    if configuration_name in ['problems', 'depends']:
                        if not isinstance(configuration_value, list):
                            err_msgs.append("{} configuration section:'{}' isn't a list.".format(cls, configuration_name))
                            continue

                        if configuration_name == 'depends':
                            err_msgs.extend(self._validate_depends(cls, configuration_value))

                        continue

                    if configuration_name == 'input_schema':
                        schema_found = True
                        if not isinstance(configuration_value, dict):
                            err_msgs.append("{} configuration section:'{}' isn't a dictionary.".format(cls, configuration_name))
                            continue

                        err_msgs.extend(self._validate_schema(cls, configuration_value))
                        continue

                if not schema_found:
                    err_msgs.append("{} configuration missing schema.".format(cls))

        if len(err_msgs) > 0:
            self.fail('\n'.join(err_msgs))

    def _validate_schema(self, cls, schema):
        properties = schema.get('properties', {})
        if not isinstance(properties, dict):
            return ["{} configuration schema '{}' isn't a dictionary.".format(cls, 'properties')]

        parameters = inspect.signature(cls.__init__).parameters
        err_msgs = []
        for prop_name, value in properties.items():
            if not isinstance(properties, dict):
                err_msgs.append("{} configuration schema '{}/{}' isn't a dictionary.".format(cls, 'properties', prop_name))
                continue

            parameter = parameters.get(prop_name)
            if parameter is None:
                # It is not mandatory that all schema parametere be in the constructor
                continue

            if 'default' in value:
                default_value = value['default']
                if parameter.default != inspect.Parameter.empty and parameter.default != default_value:
                    err_msgs.append("{} __init__ param '{}' default value '{}' different from default value '{}' "
                                    "found on its configuration schema.".format(cls, prop_name, parameter.default, default_value))
            else:
                if parameter.default != inspect.Parameter.empty:
                    err_msgs.append("{} __init__ param '{}' default value '{}' missing "
                                    "in its configuration schema.".format(cls, prop_name, parameter.default))

        return err_msgs

    def _validate_depends(self, cls, dependencies):
        err_msgs = []
        for dependency in dependencies:
            if not isinstance(dependency, dict):
                err_msgs.append("{} configuration section:'{}' item isn't a dictionary.".format(cls, 'depends'))
                continue

            dependency_pluggable_type = dependency.get('pluggable_type')
            if not isinstance(dependency_pluggable_type, str):
                err_msgs.append("{} configuration section:'{}' item:'{}' isn't a string.".format(cls, 'depends', 'pluggable_type'))
                continue

            if not any(x for x in PluggableType if x.value == dependency_pluggable_type):
                err_msgs.append("{} configuration section:'{}' item:'{}/{}' "
                                "doesn't exist.".format(cls, 'depends', 'pluggable_type', dependency_pluggable_type))
                continue

            defaults = dependency.get('default')
            if not isinstance(defaults, dict):
                continue

            default_name = defaults.get('name')
            if default_name not in local_pluggables(dependency_pluggable_type):
                print(default_name, dependency_pluggable_type, local_pluggables(dependency_pluggable_type))
                err_msgs.append("{} configuration section:'{}' item:'{}/{}/{}/{}' "
                                "not found.".format(cls, 'depends', dependency_pluggable_type, 'default', 'name', default_name))
                continue

            del defaults['name']
            if len(defaults) > 0:
                err_msgs.extend(self._validate_defaults_against_schema(dependency_pluggable_type, default_name, defaults))

        return err_msgs

    def _validate_defaults_against_schema(self, dependency_pluggable_type, default_name, defaults):
        cls = get_pluggable_class(dependency_pluggable_type, default_name)
        default_config = get_pluggable_configuration(dependency_pluggable_type, default_name)
        if not isinstance(default_config, dict):
            return ["{} configuration isn't a dictionary.".format(cls)]

        schema = default_config.get('input_schema')
        if not isinstance(default_config, dict):
            return ["{} configuration schema missing or isn't a dictionary.".format(cls)]

        properties = schema.get('properties')
        if not isinstance(properties, dict):
            return ["{} configuration schema '{}' missing or isn't a dictionary.".format(cls, 'properties')]

        err_msgs = []
        for default_property_name, default_property_value in defaults.items():
            prop = properties.get(default_property_name)
            if not isinstance(prop, dict):
                err_msgs.append("{} configuration schema '{}/{}' "
                                "missing or isn't a dictionary.".format(cls, 'properties', default_property_name))
        return err_msgs


if __name__ == '__main__':
    unittest.main()
