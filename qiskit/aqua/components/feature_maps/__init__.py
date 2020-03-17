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

"""The feature maps.

This module is deprecated, the feature maps moved to qiskit/aqua/components/ansatzes.
"""

import warnings
from qiskit.aqua.components.ansatzes.feature_maps import (PauliExpansion, PauliZExpansion,
                                                          FirstOrderExpansion, SecondOrderExpansion,
                                                          RawFeatureVector)
from .feature_map import FeatureMap

warnings.warn('The qiskit.aqua.components.feature_maps module is deprecated and will be removed '
              'no later than the release of Aqua 0.7. The feature maps '
              'are now located in qiskit.aqua.components.ansatze.feature.maps.',
              DeprecationWarning, stacklevel=2)
