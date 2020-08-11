# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Gaussian Log File Result """
import math
from typing import Dict, Union, List, Tuple, cast
import copy
import logging
import re

import numpy as np


logger = logging.getLogger(__name__)


class GaussianLogResult:
    """ Result for Gaussian™ 16 log driver.

    This result allows access to selected data from the log file that is not available
    via the use Gaussian 16 interfacing code when using the MatrixElement file.
    Since this parses the text output it is subject to the format of the log file.
    """
    def __init__(self, log: Union[str, List[str]]) -> None:
        """
        Args:
            log: The log contents conforming to Gaussian™ 16 format either as a single string
                 containing new line characters, or as a list of strings. If the single string
                 has no new line characters it is treated a file name and the file contents
                 will be read (a valid log file would be multiple lines).
        Raises:
            ValueError: Invalid Input
        """

        self._log = None

        if isinstance(log, str):
            lines = log.split('\n')

            if len(lines) == 1:
                with open(lines[0]) as file:
                    self._log = file.read().split('\n')
            else:
                self._log = lines

        elif isinstance(log, list):
            self._log = log

        else:
            raise ValueError("Invalid input for Gaussian Log Parser '{}'".format(log))

    @property
    def log(self) -> List[str]:
        """ The complete Gaussian log in the form of a list of strings. """
        return copy.copy(self._log)

    def __str__(self):
        return '\n'.join(self._log)

    # Sections of interest in the log file
    _SECTION_QUADRATIC = r':\s+QUADRATIC\sFORCE\sCONSTANTS\sIN\sNORMAL\sMODES'
    _SECTION_CUBIC = r':\s+CUBIC\sFORCE\sCONSTANTS\sIN\sNORMAL\sMODES'
    _SECTION_QUARTIC = r':\s+QUARTIC\sFORCE\sCONSTANTS\sIN\sNORMAL\sMODES'

    @property
    def quadratic_force_constants(self) -> List[Tuple[str, str, float, float, float]]:
        """ Quadratic force constants. (2 indices, 3 values)

        Returns:
            A list of tuples each with 2 index values and 3 constant values.
            An empty list is returned if no such data is present in the log.
        """
        qfc = self._force_constants(self._SECTION_QUADRATIC, 2)
        return cast(List[Tuple[str, str, float, float, float]], qfc)

    @property
    def cubic_force_constants(self) -> List[Tuple[str, str, str, float, float, float]]:
        """ Cubic force constants. (3 indices, 3 values)

        Returns:
            A list of tuples each with 3 index values and 3 constant values.
            An empty list is returned if no such data is present in the log.
        """
        cfc = self._force_constants(self._SECTION_CUBIC, 3)
        return cast(List[Tuple[str, str, str, float, float, float]], cfc)

    @property
    def quartic_force_constants(self) -> List[Tuple[str, str, str, str, float, float, float]]:
        """ Quartic force constants. (4 indices, 3 values)

        Returns:
            A list of tuples each with 4 index values and 3 constant values.
            An empty list is returned if no such data is present in the log.
        """
        qfc = self._force_constants(self._SECTION_QUARTIC, 4)
        return cast(List[Tuple[str, str, str, str, float, float, float]], qfc)

    def _force_constants(self, section_name: str, indices: int) -> List[Tuple]:
        constants = []
        pattern_constants = ''
        for i in range(indices):
            pattern_constants += r'\s+(?P<index{}>\w+)'.format(i+1)
        for i in range(3):
            pattern_constants += r'\s+(?P<const{}>[+-]?\d+\.\d+)'.format(i+1)

        # Find the section of interest
        i = 0
        found_section = False
        for i, line in enumerate(self._log):
            if re.search(section_name, line) is not None:
                found_section = True
                break

        # Now if section found look from this line onwards to get the corresponding constant data
        # lines which are from when we start to get a match against the constants pattern until we
        # do not again.
        const_found = False
        if found_section:
            for line in self._log[i:]:
                if not const_found:
                    # If we have not found the first line that matches we keep looking
                    # until we get a match (non-None) and then drop through into found
                    # section which we use thereafter
                    const = re.match(pattern_constants, line)
                    const_found = const is not None

                if const_found:
                    # If we found the match then for each line we want the contents until
                    # such point as it does not match anymore then we break out
                    const = re.match(pattern_constants, line)
                    if const is not None:
                        clist = []  # type: List[Union[str, float]]
                        for i in range(indices):
                            clist.append(const.group('index{}'.format(i + 1)))
                        for i in range(3):
                            clist.append(float(const.group('const{}'.format(i + 1))))
                        constants.append(tuple(clist))
                    else:
                        break   # End of matching lines

        return constants

    @property
    def a_to_h_numbering(self) -> Dict[str, int]:
        """ A to H numbering mapping.

        Returns:
            Dictionary mapping string A numbering such as '1', '3a' etc from forces modes
            to H integer numbering
        """
        a2h = {}  # type: Dict[str, int]

        found_section = False
        found_h = False
        found_a = False
        for line in self._log:
            if not found_section:
                if re.search(r'Input/Output\sinformation', line) is not None:
                    logger.debug(line)
                    found_section = True
            else:
                if re.search(r'\s+\(H\)\s+\|', line) is not None:
                    logger.debug(line)
                    found_h = True
                    h_nums = [x.strip() for x in line.split('|') if x and '(H)' not in x]
                elif re.search(r'\s+\(A\)\s+\|', line) is not None:
                    logger.debug(line)
                    found_a = True
                    a_nums = [x.strip() for x in line.split('|') if x and '(A)' not in x]

                if found_h and found_a:
                    for i, a_num in enumerate(a_nums):
                        a2h[a_num] = int(h_nums[i])
                    break

        return a2h

    # ----------------------------------------------------------------------------------------
    # The following is to process the constants and produce an n-body array for input
    # to the Bosonic Operator. It maybe these methods all should be in some other module
    # but for now they are here

    @staticmethod
    def _multinomial(indices: List[int]) -> float:
        # For a given list of integers, computes the associated multinomial
        tmp = set(indices)  # Set of uniques indices
        multinomial = 1
        for i, val in enumerate(tmp):
            count = indices.count(val)
            multinomial = multinomial * math.factorial(count)
        return multinomial

    def _process_entry_indices(self, entry: List[Union[str, float]]) -> List[int]:
        # a2h gives us say '3a' -> 1, '3b' -> 2 etc. The H values can be 1 thru 4
        # but we want them numbered in reverse order so the 'a2h_vals + 1 - a2h[x]'
        # takes care of this
        a2h = self.a_to_h_numbering
        a2h_vals = max(list(a2h.values()))

        num_indices = len(entry) - 3
        return [a2h_vals + 1 - a2h[x] for x in entry[0:num_indices]]

    def _compute_modes(self, normalize: bool = True) -> List[List[Union[int, float]]]:
        # Returns [value, idx0, idx1...] from 2 indices (quadratic) to 4 (quartic)
        qua = self.quadratic_force_constants
        cub = self.cubic_force_constants
        qrt = self.quartic_force_constants
        modes = []
        for entry in qua:
            indices = self._process_entry_indices(list(entry))
            if indices:
                factor = 2.0
                factor *= self._multinomial(indices) if normalize else 1.0
                line = [entry[2] / factor]
                line.extend(indices)
                modes.append(line)
                modes.append([-x for x in line])
        for entry in cub:
            indices = self._process_entry_indices(list(entry))
            if indices:
                factor = 2.0 * math.sqrt(2.0)
                factor *= self._multinomial(indices) if normalize else 1.0
                line = [entry[3] / factor]
                line.extend(indices)
                modes.append(line)
        for entry in qrt:
            indices = self._process_entry_indices(list(entry))
            if indices:
                factor = 4.0
                factor *= self._multinomial(indices) if normalize else 1.0
                line = [entry[4] / factor]
                line.extend(indices)
                modes.append(line)

        return modes

    @staticmethod
    def _harmonic_integrals(m: int, n: int, power: int, omega: int):
        coeff = 0
        if power == 1:
            if abs(n - m) == 1:
                coeff = np.sqrt(n / (2 * omega))
        elif power == 2:
            if abs(n - m) == 0:
                coeff = (n + 1 / 2) / omega
            elif abs(n - m) == 2:
                coeff = np.sqrt(n * (n - 1)) / (2 * omega)
        elif power == 3:
            if abs(n - m) == 1:
                coeff = 3 * np.power(n / (2 * omega), 3 / 2)
            elif abs(n - m) == 3:
                coeff = np.sqrt(n * (n - 1) * (n - 2)) / np.power(2 * omega, 3 / 2)
        elif power == 4:
            if abs(n - m) == 0:
                coeff = (6 * n * (n + 1) + 3) / (4 * omega ** 2)
            elif abs(n - m) == 2:
                coeff = (n - 1 / 2) * np.sqrt(n * (n - 1)) / omega ** 2
            elif abs(n - m) == 4:
                coeff = np.sqrt(n * (n - 1) * (n - 2) * (n - 3)) / (4 * omega ** 2)
        else:
            raise ValueError('The expansion order of the PES is too high.')
        return coeff

    def compute_harmonic_modes(self, threshold=1e-6):
        omega = {1: 1, 2: 1, 3: 1, 4: 1}
        # num_modes = 4  # unused
        num_modals = 3

        harmonics = []

        entries = self._compute_modes()
        for entry in entries:
            coeff0 = entry[0]
            indices = entry[1:]

            # Note: these negative indices as detected below are explicitly generated in
            # _compute_modes for other potential uses. They are not wanted by this logic.
            if any([index < 0 for index in indices]):
                continue
            indexes = {}  # type: Dict[int, int]
            for i in indices:
                if indexes.get(i) is None:
                    indexes[i] = 1
                else:
                    indexes[i] += 1

            order = len(indexes.keys())
            modes = list(indexes.keys())

            if order == 1:
                for m in range(num_modals):
                    for n in range(num_modals):
                        coeff = coeff0 * self._harmonic_integrals(m, n, indexes[modes[0]],
                                                                  omega[modes[0]])
                        if abs(coeff) > threshold:
                            harmonics.append([modes[0], n, m, coeff])

            elif order == 2:
                for m in range(num_modals):
                    for n in range(num_modals):
                        coeff = coeff0 * self._harmonic_integrals(m, n, indexes[modes[0]],
                                                                  omega[modes[0]])
                        for j in range(num_modals):
                            for k in range(num_modals):
                                coeff *= self._harmonic_integrals(j, k, indexes[modes[1]],
                                                                  omega[modes[1]])
                                if abs(coeff) > threshold:
                                    harmonics.append([modes[0], n, m,
                                                      modes[1], j, k, coeff])
            elif order == 3:
                for m in range(num_modals):
                    for n in range(num_modals):
                        coeff = coeff0 * self._harmonic_integrals(m, n, indexes[modes[0]],
                                                                  omega[modes[0]])
                        for j in range(num_modals):
                            for k in range(num_modals):
                                coeff *= self._harmonic_integrals(j, k, indexes[modes[1]],
                                                                  omega[modes[1]])
                                for p in range(num_modals):
                                    for q in range(num_modals):
                                        coeff *= self._harmonic_integrals(p, q, indexes[modes[2]],
                                                                          omega[modes[2]])
                                        if abs(coeff) > threshold:
                                            harmonics.append([modes[0], n, m,
                                                              modes[1], j, k,
                                                              modes[2], p, q, coeff])
            else:
                raise ValueError('Unexpected order value of {}'.format(order))

        return harmonics
