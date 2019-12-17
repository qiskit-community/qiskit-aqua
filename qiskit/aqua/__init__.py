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
===============================================================
Aqua (Algorithms for QUantum Applications) (:mod:`qiskit.aqua`)
===============================================================
Qiskit Aqua provides a library of quantum algorithms and components
to build quantum applications and leverage near-term devices.

.. currentmodule:: qiskit.aqua

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   aqua_globals

Aqua globals class provides random number and max parallel process configuration.
Aqua uses the random function and max parallel processes when running any
function requiring randomization and/or that can be be done in parallel. Setting
the random seed to a given value will ensure predictability in outcome when using
a simulator (seeds should also be set in :class:`QuantumInstance` for transpiler
and simulator too).

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   AquaError

In addition to standard Python errors Aqua will raise this error if circumstances
are that it cannot proceed to completion.

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   QuantumInstance

A QuantumInstance holds the Qiskit `backend` as well as a number of compile and
runtime parameters controlling circuit compilation and execution. Aqua's quantum
:mod:`algorithms <qiskit.aqua.algorithms>`
are run on a device or simulator by passing a QuantumInstance setup with the desired
backend etc.

Submodules
==========

.. autosummary::
   :toctree:

   algorithms
   components
   circuits
   operators
   utils

"""


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
