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
Profiler decorator
"""

import logging
from qiskit.aqua._logging import add_qiskit_aqua_logging_level

logger = logging.getLogger(__name__)

# Add extra levels of logging for performance analysis
add_qiskit_aqua_logging_level('MPROFILE', logging.DEBUG - 1)
add_qiskit_aqua_logging_level('CPROFILE', logging.DEBUG - 2)
add_qiskit_aqua_logging_level('LPROFILE', logging.DEBUG - 3)
add_qiskit_aqua_logging_level('MEMPROFILE', logging.DEBUG - 4)


class QProfile:

    def __init__(self, function):
        self.function = function

    def __call__(self, *args, **kwargs):
        if logger.getEffectiveLevel() == logging.MPROFILE:
            import sys
            result = self.function(*args, **kwargs)

            logger.debug(">>Tracing memory size : qobj is %s bytes",
                         sys.getsizeof(result))
            return result

        elif logger.getEffectiveLevel() == logging.CPROFILE:
            import cProfile
            profile = cProfile.Profile()
            try:
                profile.enable()
                result = self.function(*args, **kwargs)
                profile.disable()
                return result
            finally:
                profile.print_stats()

        elif logger.getEffectiveLevel() == logging.LPROFILE:
            try:
                from line_profiler import LineProfiler
                try:
                    profiler = LineProfiler()
                    profiler.add_function(self.function)

                    profiler.enable_by_count()
                    result = self.function(*args, **kwargs)
                    return result
                finally:
                    profiler.print_stats()
            except ImportError:
                return self.function(*args, **kwargs)

        elif logger.getEffectiveLevel() == logging.MEMPROFILE:
            try:
                from memory_profiler import LineProfiler, show_results
                try:
                    profiler = LineProfiler()
                    profiler.add_function(self.function)
                    profiler.enable_by_count()
                    result = self.function(*args, **kwargs)
                    return result
                finally:
                    show_results(profiler)
            except ImportError:
                return self.function(*args, **kwargs)
        else:
            return self.function(*args, **kwargs)
