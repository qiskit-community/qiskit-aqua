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

import os
import unittest
import importlib
import sys
import argparse


def get_all_test_modules():
    """
    Gathers all test modules
    """
    test_modules = []
    current_directory = os.path.dirname(__file__)
    sys.path.insert(0, os.path.join(current_directory, '..'))
    files = sorted(os.listdir(current_directory))
    for file in files:
        if file.startswith('test') and file.endswith('.py'):
            test_modules.append(file[:-3])

    return test_modules


class CustomTests():
    """
    Lists sets of chosen tests
    """

    def __init__(self, test_modules):
        self.test_modules = test_modules

    def suite(self):
        alltests = unittest.TestSuite()
        for name in self.test_modules:
            module = importlib.import_module(name, package=None)
            alltests.addTest(unittest.findTestCases(module))

        return alltests


def check_positive(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(
            '{} is an invalid positive int value'.format(value))
    return ivalue


def check_positive_or_zero(value):
    ivalue = int(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError(
            '{} is an invalid positive int or zero value'.format(value))
    return ivalue


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Qiskit Aqua Unit Test Tool')
    parser.add_argument('start',
                        metavar='start',
                        type=check_positive_or_zero,
                        help='start index of test modules to run')
    parser.add_argument('-end',
                        metavar='index',
                        type=check_positive,
                        help='end index of test modules to run',
                        required=False)

    args = parser.parse_args()

    test_modules = get_all_test_modules()
    tests_count = len(test_modules)
    if tests_count == 0:
        raise Exception('No test modules found.')

    start_index = args.start
    if start_index >= tests_count:
        raise Exception('Start index {} >= number of test modules {}.'.format(
            start_index, tests_count))

    end_index = args.end if args.end is not None else tests_count
    if start_index >= end_index:
        raise Exception('Start index {} >= end index {}.'.format(
            start_index, end_index))

    customTests = CustomTests(test_modules[start_index:end_index])
    unittest.main(argv=['first-arg-is-ignored'],
                  defaultTest='customTests.suite', verbosity=2)
