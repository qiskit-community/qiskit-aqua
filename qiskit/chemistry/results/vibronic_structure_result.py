# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The vibronic structure result."""

import logging

from .eigenstate_result import EigenstateResult

logger = logging.getLogger(__name__)


class VibronicStructureResult(EigenstateResult):
    """The vibronic structure result."""
    # TODO: just a placeholder to pass lint, needs to be defined
