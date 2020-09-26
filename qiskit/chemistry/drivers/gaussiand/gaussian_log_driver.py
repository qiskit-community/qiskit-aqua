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

""" Gaussian Log Driver """

from typing import Union, List
import logging

from qiskit.chemistry import QiskitChemistryError
from .gaussian_utils import check_valid, run_g16
from .gaussian_log_result import GaussianLogResult

logger = logging.getLogger(__name__)


class GaussianLogDriver:
    """  Gaussian™ 16 log driver.

    Qiskit chemistry driver using the Gaussian™ 16 program that provides the log
    back, via :class:`GaussianLogResult`, for access to the log and data recorded there.

    See http://gaussian.com/gaussian16/

    This driver does not use Gaussian 16 interfacing code, as certain data such as forces
    properties are not present in the MatrixElement file. The log is returned as a
    :class:`GaussianLogResult` allowing it to be parsed for whatever data may be of interest.
    This result class also contains ready access to certain data within the log.
    """

    def __init__(self, jcf: Union[str, List[str]]) -> None:
        r"""
        Args:
            jcf: A job control file conforming to Gaussian™ 16 format. This can
                be provided as a single string with '\\n' line separators or as a list of
                strings.
        Raises:
            QiskitChemistryError: Invalid Input
        """
        GaussianLogDriver._check_valid()

        if not isinstance(jcf, list) and not isinstance(jcf, str):
            raise QiskitChemistryError("Invalid input for Gaussian Log Driver '{}'"
                                       .format(jcf))

        if isinstance(jcf, list):
            jcf = '\n'.join(jcf)

        self._jcf = jcf

    @staticmethod
    def _check_valid():
        check_valid()

    def run(self) -> GaussianLogResult:
        """ Runs the driver to produce a result given the supplied job control file.

        Returns:
            A log file result.

        Raises:
            QiskitChemistryError: Missing output log
        """
        # The job control file, needs to end with a blank line to be valid for
        # Gaussian to process it. We simply add the blank line here if not.
        cfg = self._jcf
        while not cfg.endswith('\n\n'):
            cfg += '\n'

        logger.debug("User supplied job control file raw: '%s'",
                     cfg.replace('\r', '\\r').replace('\n', '\\n'))
        logger.debug('User supplied job control file\n%s', cfg)

        all_text = run_g16(cfg)
        if not all_text:
            raise QiskitChemistryError("Failed to capture log from stdout")

        return GaussianLogResult(all_text)
