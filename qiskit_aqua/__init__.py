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

"""Algorithm discovery methods, Error and Base classes"""

from .utils import cnx
from .algorithmerror import AlgorithmError
from .operator import Operator
from .preferences import Preferences
from .quantumalgorithm import QuantumAlgorithm
from ._discover import (refresh_pluggables,
                        local_pluggables_types,
                        local_pluggables,
                        get_pluggable_configuration)


__version__ = '0.3.1'

__all__ = ['AlgorithmError',
           'Operator',
           'Preferences',
           'QuantumAlgorithm',
           'refresh_pluggables',
           'local_pluggables_types',
           'local_pluggables',
           'get_pluggable_configuration',
           'run_algorithm',
           'run_algorithm_to_json']

from ._discover import _PLUGGABLES

prefix = 'from ._discover import '
for pluggable_type in _PLUGGABLES.keys():
    method = 'register_{}'.format(pluggable_type)
    exec(prefix + method)
    __all__.append(method)
    method = 'deregister_{}'.format(pluggable_type)
    exec(prefix + method)
    __all__.append(method)
    method = 'get_{}_class'.format(pluggable_type)
    exec(prefix + method)
    __all__.append(method)
    method = 'get_{}_instance'.format(pluggable_type)
    exec(prefix + method)
    __all__.append(method)
    method = 'get_{}_configuration'.format(pluggable_type)
    exec(prefix + method)
    __all__.append(method)
    method = 'local_{}s'.format(pluggable_type)
    exec(prefix + method)
    __all__.append(method)

from .algomethods import run_algorithm, run_algorithm_to_json
