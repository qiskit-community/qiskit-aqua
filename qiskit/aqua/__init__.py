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
from .qiskit_aqua_globals import aqua_globals
from .preferences import Preferences
from ._discover import (PLUGGABLES_ENTRY_POINT,
                        PluggableType,
                        refresh_pluggables,
                        local_pluggables_types,
                        local_pluggables,
                        get_pluggable_class,
                        get_pluggable_configuration,
                        register_pluggable,
                        deregister_pluggable)
from .utils.backend_utils import (get_aer_backend,
                                  get_backends_from_provider,
                                  get_backend_from_provider,
                                  get_local_providers,
                                  register_ibmq_and_get_known_providers,
                                  get_provider_from_backend,
                                  enable_ibmq_account,
                                  disable_ibmq_account)
from .pluggable import Pluggable
from .quantum_instance import QuantumInstance
from .operator import Operator
from .algorithms import QuantumAlgorithm
from .qiskit_aqua import (QiskitAqua,
                          execute_algorithm,
                          run_algorithm,
                          run_algorithm_to_json)
from ._logging import (get_logging_level,
                       build_logging_config,
                       set_logging_config,
                       get_qiskit_aqua_logging,
                       set_qiskit_aqua_logging)

__version__ = '0.4.2'

__all__ = ['AquaError',
           'Preferences',
           'Pluggable',
           'Operator',
           'QuantumAlgorithm',
           'PLUGGABLES_ENTRY_POINT',
           'PluggableType',
           'refresh_pluggables',
           'QuantumInstance',
           'get_aer_backend',
           'get_backends_from_provider',
           'get_backend_from_provider',
           'get_local_providers',
           'register_ibmq_and_get_known_providers',
           'get_provider_from_backend',
           'enable_ibmq_account',
           'disable_ibmq_account',
           'local_pluggables_types',
           'local_pluggables',
           'get_pluggable_class',
           'get_pluggable_configuration',
           'register_pluggable',
           'deregister_pluggable',
           'aqua_globals',
           'QiskitAqua',
           'execute_algorithm',
           'run_algorithm',
           'run_algorithm_to_json',
           'get_logging_level',
           'build_logging_config',
           'set_logging_config',
           'get_qiskit_aqua_logging',
           'set_qiskit_aqua_logging']
