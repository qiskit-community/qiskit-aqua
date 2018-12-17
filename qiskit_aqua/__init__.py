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

from .aqua_error import AquaError
from ._discover import (PluggableType,
                        refresh_pluggables,
                        local_pluggables_types,
                        local_pluggables,
                        get_pluggable_class,
                        get_pluggable_configuration,
                        register_pluggable,
                        deregister_pluggable)
from .pluggable import Pluggable
from .utils.backend_utils import get_aer_backend, get_aer_backends
from .utils.cnx import cnx
from .quantum_instance import QuantumInstance
from .operator import Operator
from .algorithms import QuantumAlgorithm
from ._aqua import run_algorithm, run_algorithm_to_json
from ._logging import (get_logging_level,
                       build_logging_config,
                       set_logging_config,
                       get_aqua_logging,
                       set_aqua_logging)

__version__ = '0.4.0'

__all__ = ['AquaError',
           'Pluggable',
           'Operator',
           'QuantumAlgorithm',
           'PluggableType',
           'refresh_pluggables',
           'QuantumInstance',
           'get_aer_backend',
           'get_aer_backends',
           'local_pluggables_types',
           'local_pluggables',
           'get_pluggable_class',
           'get_pluggable_configuration',
           'register_pluggable',
           'deregister_pluggable',
           'run_algorithm',
           'run_algorithm_to_json',
           'get_logging_level',
           'build_logging_config',
           'set_logging_config',
           'get_aqua_logging',
           'set_aqua_logging']
