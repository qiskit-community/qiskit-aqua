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
"""Algorithm discovery methods, Error and Base classes"""

from .version import __version__
from .aqua_error import AquaError
from .aqua_globals import aqua_globals
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
                                  get_provider_from_backend)
from .pluggable import Pluggable
from .quantum_instance import QuantumInstance
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

__all__ = ['__version__',
           'AquaError',
           'Preferences',
           'Pluggable',
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
