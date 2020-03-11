# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


class QiskitOptimizationError(Exception):
    """Class for errors returned by Qiskit Optimization libraries functions.

    self.args[0] : A string describing the error.
    """

    def __str__(self):
        # Note: this is actually ok. Exception does have subscriptable args.
        # pylint: disable=unsubscriptable-object
        return self.args[0]
