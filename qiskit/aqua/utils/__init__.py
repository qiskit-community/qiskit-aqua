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

""" Utility packages """

from .tensor_product import tensorproduct
from .json_utils import convert_dict_to_json, convert_json_to_dict
from .random_matrix_generator import (random_unitary, random_h2_body,
                                      random_h1_body, random_hermitian,
                                      random_non_hermitian)
from .decimal_to_binary import decimal_to_binary
from .circuit_utils import summarize_circuits
from .subsystem import get_subsystem_density_matrix, get_subsystems_counts
from .entangler_map import get_entangler_map, validate_entangler_map
from .dataset_helper import (get_feature_dimension, get_num_classes,
                             split_dataset_to_data_and_labels,
                             map_label_to_class_name, reduce_dim_to_via_pca)
from .qp_solver import optimize_svm
from .circuit_factory import CircuitFactory
from .backend_utils import has_ibmq, has_aer

__all__ = [
    'tensorproduct',
    'convert_dict_to_json',
    'convert_json_to_dict',
    'random_unitary',
    'random_h2_body',
    'random_h1_body',
    'random_hermitian',
    'random_non_hermitian',
    'decimal_to_binary',
    'summarize_circuits',
    'get_subsystem_density_matrix',
    'get_subsystems_counts',
    'get_entangler_map',
    'validate_entangler_map',
    'get_feature_dimension',
    'get_num_classes',
    'split_dataset_to_data_and_labels',
    'map_label_to_class_name',
    'reduce_dim_to_via_pca',
    'optimize_svm',
    'CircuitFactory',
    'has_ibmq',
    'has_aer',
]
