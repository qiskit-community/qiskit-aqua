# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""
Exception for errors raised by Aqua.
"""


class AquaError(Exception):
    """Base class for errors raised by Aqua."""

    def __init__(self, *message):
        """Set the error message."""
        super(AquaError, self).__init__(' '.join(message))
        self.message = ' '.join(message)

    def __str__(self) -> str:
        """Return the message."""
        return repr(self.message)
