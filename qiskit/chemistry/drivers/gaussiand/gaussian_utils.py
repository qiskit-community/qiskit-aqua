# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Gaussian Utility Methods """

import logging
from subprocess import Popen, PIPE
from shutil import which
from qiskit.chemistry import QiskitChemistryError

logger = logging.getLogger(__name__)

_GAUSSIAN_16 = 'g16'
_GAUSSIAN_16_DESC = 'Gaussian 16'

_G16PROG = which(_GAUSSIAN_16)


def check_valid() -> None:
    """ Checks if Gaussian is installed and available"""
    if _G16PROG is None:
        raise QiskitChemistryError(
            "Could not locate {} executable '{}'. Please check that it is installed correctly."
            .format(_GAUSSIAN_16_DESC, _GAUSSIAN_16))


def run_g16(cfg: str) -> str:
    """
    Runs Gaussian 16. We capture stdout and if error log the last 10 lines that
    should include the error description from Gaussian.

    Args:
        cfg: configuration

    Returns:
        Text log output

    Raises:
        QiskitChemistryError: Failed run or log not captured.

    """
    process = None
    try:
        with Popen(_GAUSSIAN_16, stdin=PIPE, stdout=PIPE, universal_newlines=True) as process:
            stdout, _ = process.communicate(cfg)
            process.wait()
    except Exception as ex:
        if process is not None:
            process.kill()

        raise QiskitChemistryError('{} run has failed'.format(_GAUSSIAN_16_DESC)) from ex

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
                _GAUSSIAN_16_DESC, process.returncode, errmsg))

    all_text = ""
    if stdout is not None:
        lines = stdout.splitlines()
        for line in lines:
            all_text += line + "\n"

    logger.debug("Gaussian output:\n%s", all_text)

    return all_text
