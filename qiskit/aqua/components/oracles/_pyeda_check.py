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
"""
Check existence of pyeda.
"""

import importlib
import logging
from qiskit.aqua import AquaError

logger = logging.getLogger(__name__)


def _check_pluggable_valid(name):
    err_msg = "Unable to instantiate '{}', pyeda is not installed. Please look at https://pyeda.readthedocs.io/en/latest/install.html.".format(name)
    try:
        spec = importlib.util.find_spec('pyeda')
        if spec is not None:
            return
    except Exception as e:
        logger.debug('{} {}'.format(err_msg, str(e)))
        raise AquaError(err_msg) from e

    raise AquaError(err_msg)
