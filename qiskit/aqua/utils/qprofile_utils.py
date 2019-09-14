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

# pylint: skip-file

"""
Profiler decorator
"""

import logging

from qiskit.aqua._logging import add_qiskit_aqua_logging_level

logger = logging.getLogger(__name__)
add_qiskit_aqua_logging_level('MPROFILE', logging.DEBUG - 6)
logging.getLogger(__name__).setLevel("MPROFILE")

add_qiskit_aqua_logging_level('CPROFILE', logging.MPROFILE - 1)
logging.getLogger(__name__).setLevel("CPROFILE")

add_qiskit_aqua_logging_level('MEMPROFILE', logging.MPROFILE - 2)
logging.getLogger(__name__).setLevel("MEMPROFILE")

if logger.getEffectiveLevel() >= logging.MPROFILE:

    def qprofile(func):
        def wrapper(*original_args, **original_kwargs):
            qobj = func(*original_args, **original_kwargs)
            if logger.getEffectiveLevel() >= logging.MPROFILE:
                import sys
                logger.debug(">>Tracing memory size : qobj is %s bytes", sys.getsizeof(qobj))
            return qobj
        return wrapper
elif logger.getEffectiveLevel() >= logging.CPROFILE:
    try:
        from line_profiler import LineProfiler

        def qprofile(func):
            def wrapper(*original_args, **original_kwargs):
                try:
                    profiler = LineProfiler()
                    profiler.add_function(func)

                    profiler.enable_by_count()
                    return func(*original_args, **original_kwargs)
                finally:
                    profiler.print_stats()
            return wrapper
    except ImportError:
        def qprofile(func):
            """
            Helpful if you accidentally leave in production!
            """
            def wrapper(*original_args, **original_kwargs):
                return func(*original_args, **original_kwargs)
            return wrapper

elif logger.getEffectiveLevel() >= logging.MEMPROFILE:

    try:
        from memory_profiler import LineProfiler, show_results

        def qprofile(func):

            def wrapper(*original_args, **original_kwargs):
                try:
                    profiler = LineProfiler()
                    profiler.add_function(func)
                    profiler.enable_by_count()
                    return func(*original_args, **original_kwargs)
                finally:
                    show_results(profiler)
            return wrapper
    except ImportError:
        def qprofile(func):
            """
            Helpful if you accidentally leave in production!
            """
            def wrapper(*original_args, **original_kwargs):
                return func(*original_args, **original_kwargs)
            return wrapper
