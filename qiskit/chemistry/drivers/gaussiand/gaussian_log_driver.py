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
from subprocess import Popen, PIPE
from shutil import which

from qiskit.chemistry import QiskitChemistryError
from .gaussiandriver import GAUSSIAN_16, GAUSSIAN_16_DESC
from .gaussian_log_result import GaussianLogResult

logger = logging.getLogger(__name__)

G16PROG = which(GAUSSIAN_16)


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
        if G16PROG is None:
            raise QiskitChemistryError(
                "Could not locate {} executable '{}'. Please check that it is installed correctly."
                .format(GAUSSIAN_16_DESC, GAUSSIAN_16))

    def run(self) -> GaussianLogResult:
        """ Runs the driver to produce a result given the supplied job control file.

        Returns:
            A log file result.
        """
        # The job control file, needs to end with a blank line to be valid for
        # Gaussian to process it. We simply add the blank line here if not.
        cfg = self._jcf
        while not cfg.endswith('\n\n'):
            cfg += '\n'

        logger.debug("User supplied job control file raw: '%s'",
                     cfg.replace('\r', '\\r').replace('\n', '\\n'))
        logger.debug('User supplied job control file\n%s', cfg)

        return GaussianLogDriver._run_g16(cfg)

    @staticmethod
    def _run_g16(cfg: str) -> GaussianLogResult:

        # Run Gaussian 16. We capture stdout and if error log the last 10 lines that
        # should include the error description from Gaussian
        process = None
        try:
            process = Popen(GAUSSIAN_16, stdin=PIPE, stdout=PIPE, universal_newlines=True)
            stdout, _ = process.communicate(cfg)
            process.wait()
        except Exception as ex:
            if process is not None:
                process.kill()

            raise QiskitChemistryError('{} run has failed'.format(GAUSSIAN_16_DESC)) from ex

        if process.returncode != 0:
            errmsg = ""
            if stdout is not None:
                lines = stdout.splitlines()
                start = 0
                if len(lines) > 10:
                    start = len(lines) - 10
                for i in range(start, len(lines)):
                    logger.error(lines[i])
                    errmsg += lines[i] + "\n"
            raise QiskitChemistryError(
                '{} process return code {}\n{}'.format(
                    GAUSSIAN_16_DESC, process.returncode, errmsg))

        alltext = ""
        if stdout is not None:
            lines = stdout.splitlines()
            for line in lines:
                alltext += line + "\n"

        if not alltext:
            raise QiskitChemistryError("Failed to capture log from stdout")

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Gaussian output:\n%s", alltext)

        return GaussianLogResult(alltext)
