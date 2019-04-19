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
