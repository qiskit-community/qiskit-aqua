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

""" Gaussian Log File Parser """

from typing import Union, List, Tuple
import copy
import logging
import re

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
                 has no new line characters it is treated a file name and will be read (a valid
                 log file contents is multiple lines).
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
        """ Gets the complete log in the form of a list of strings """
        return copy.copy(self._log)

    def __str__(self):
        return '\n'.join(self._log)

    # Sections of interest in the log file
    SECTION_QUADRATIC = r':\s+QUADRATIC\sFORCE\sCONSTANTS\sIN\sNORMAL\sMODES'
    SECTION_CUBIC = r':\s+CUBIC\sFORCE\sCONSTANTS\sIN\sNORMAL\sMODES'
    SECTION_QUARTIC = r':\s+QUARTIC\sFORCE\sCONSTANTS\sIN\sNORMAL\sMODES'

    @property
    def quadratic_force_constants(self) -> List[Tuple[str, str, float, float, float]]:
        """ Quadratic force constants: 2 indices and 3 constant values.
            An empty list is returned if no such data is present in the log.
        """
        return self._force_constants(self.SECTION_QUADRATIC, 2)

    @property
    def cubic_force_constants(self) -> List[Tuple[str, str, str, float, float, float]]:
        """ Cubic force constants: 3 indices and 3 constant values.
            An empty list is returned if no such data is present in the log.
        """
        return self._force_constants(self.SECTION_CUBIC, 3)

    @property
    def quartic_force_constants(self) -> List[Tuple[str, str, str, str, float, float, float]]:
        """ Cubic force constants: 4 indices and 3 constant values.
            An empty list is returned if no such data is present in the log.
        """
        return self._force_constants(self.SECTION_QUARTIC, 4)

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

        # Now if section found look from this point on to get the corresponding constant data lines
        # which is from when we start to get a match against the constants pattern until we
        # do not again.
        const_found = False
        if found_section:
            for j, line in enumerate(self._log[i:]):
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
                        clist = []
                        for i in range(indices):
                            clist.append(const.group('index{}'.format(i + 1)))
                        for i in range(3):
                            clist.append(float(const.group('const{}'.format(i + 1))))
                        constants.append(tuple(clist))
                    else:
                        break   # End of matching lines

        return constants



