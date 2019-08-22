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

""" Custom Tests Tool """

import os
import unittest
import importlib
import sys
import argparse


def _get_all_test_modules(folder):
    """
    Gathers all test modules
    """
    modules = []
    current_directory = os.path.dirname(__file__)
    sys.path.insert(0, os.path.join(current_directory, '..'))
    test_directory = os.path.join(current_directory, folder) if folder else current_directory
    for dirpath, _, filenames in os.walk(test_directory):
        module = os.path.relpath(dirpath, current_directory).replace('/', '.')
        for file in filenames:
            if file.startswith('test') and file.endswith('.py'):
                modules.append('{}.{}'.format(module, file[:-3]))

    return sorted(modules)


class CustomTests():
    """
    Lists sets of chosen tests
    """

    def __init__(self, modules):
        self.modules = modules

    def suite(self):
        """ test suite """
        alltests = unittest.TestSuite()
        for name in self.modules:
            module = importlib.import_module(name, package=None)
            alltests.addTest(unittest.findTestCases(module))

        return alltests


def _check_positive(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(
            '{} is an invalid positive int value'.format(value))
    return ivalue


def _check_positive_or_zero(value):
    ivalue = int(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError(
            '{} is an invalid positive int or zero value'.format(value))
    return ivalue


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(description='Qiskit Aqua Unit Test Tool')
    PARSER.add_argument('-dir',
                        metavar='dir',
                        help='relative folder from test with modules',
                        required=False)
    PARSER.add_argument('-start',
                        metavar='start',
                        type=_check_positive_or_zero,
                        help='start index of test modules to run',
                        required=False)
    PARSER.add_argument('-end',
                        metavar='index',
                        type=_check_positive,
                        help='end index of test modules to run',
                        required=False)

    ARGS = PARSER.parse_args()

    TEST_MODULES = _get_all_test_modules(ARGS.dir)
    TESTS_COUNT = len(TEST_MODULES)
    if TESTS_COUNT == 0:
        raise Exception('No test modules found.')

    # for index, test_module in enumerate(test_modules):
    #    print(index, test_module)

    # print('Total modules:', tests_count)
    START_INDEX = ARGS.start if ARGS.start is not None else 0
    if START_INDEX >= TESTS_COUNT:
        raise Exception('Start index {} >= number of test modules {}.'.format(
            START_INDEX, TESTS_COUNT))

    END_INDEX = ARGS.end if ARGS.end is not None else TESTS_COUNT
    if START_INDEX >= END_INDEX:
        raise Exception('Start index {} >= end index {}.'.format(
            START_INDEX, END_INDEX))

    CUSTOM_TESTS = CustomTests(TEST_MODULES[START_INDEX:END_INDEX])
    unittest.main(argv=['first-arg-is-ignored'],
                  defaultTest='CUSTOM_TESTS.suite', verbosity=2)
