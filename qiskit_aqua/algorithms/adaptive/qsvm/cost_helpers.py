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

import numpy as np


def assign_label(measured_key, num_classes):
    """
    Classes = 2:
    - If odd number of qubits we use majority vote
    - If even number of qubits we use parity
    Classes = 3
    - We use part-parity
        {ex. for 2 qubits: [00], [01,10], [11] would be the three labels}
    Args:
        measured_key (str): measured key
        num_classes (int): number of classes
    """
    measured_key = np.asarray([int(k) for k in list(measured_key)])
    num_qubits = len(measured_key)
    if num_classes == 2:
        if num_qubits % 2 != 0:
            total = np.sum(measured_key)
            return 1 if total > num_qubits / 2 else 0
        else:
            hamming_weight = np.sum(measured_key)
            is_odd_parity = hamming_weight % 2
            return is_odd_parity

    elif num_classes == 3:
        first_half = int(np.floor(num_qubits / 2))
        modulo = num_qubits % 2
        # First half of key
        hamming_weight_1 = np.sum(measured_key[0:first_half + modulo])
        # Second half of key
        hamming_weight_2 = np.sum(measured_key[first_half + modulo:])
        is_odd_parity_1 = hamming_weight_1 % 2
        is_odd_parity_2 = hamming_weight_2 % 2

        return is_odd_parity_1 + is_odd_parity_2

    else:
        total_size = 2**num_qubits
        class_step = np.floor(total_size / num_classes)

        decimal_value = measured_key.dot(1 << np.arange(measured_key.shape[-1] - 1, -1, -1))
        key_order = int(decimal_value / class_step)
        return key_order if key_order < num_classes else num_classes - 1


def cost_estimate_sigmoid(shots, probs, gt_labels):
    """Calculate sigmoid cross entropy over the predicted probs
    p is the prob of gt_label.
    For class = 2:
    if p ~= 1.0, loss = 1.0
    elif p ~= 0.0, loss = 0.0
    else, x = sqrt(shots) * (0.5 - p) / sqrt(2 * p * (1-p))
            loss = 1 / (1 + exp(-x))
    For class = 3:
    if p ~= 1.0, loss = 1.0
    elif p ~= 0.0, loss = 0.0
    else, x = sqrt(shots) * ((1 +|p_0 - p_1|)/3 - p) / sqrt(2 * p * (1-p))
            loss = 1 / (1 + exp(-x))
    Args:
        shots (int): the number of shots used in quantum computing
        probs (numpy.ndarray): NxK array, N is the number of data and K is the number of class
        gt_labels (numpy.ndarray): Nx1 array
    Returns:
        float: averaged sigmoid cross entropy loss between estimated probs and gt_labels
    """
    p = probs[np.arange(0, gt_labels.shape[0]), gt_labels]
    p = np.clip(p, 0.0, 1.0)
    number_of_classes = probs.shape[1]
    loss = np.zeros(p.shape[0])
    all_index = np.arange(0, p.shape[0])
    zero_index = np.where(np.isclose(p, 0.0) is True)
    one_index = np.where(np.isclose(p, 1.0) is True)
    other_index = np.setdiff1d(all_index, np.concatenate((zero_index, one_index)))
    rest_p = p[other_index]
    denominator = np.sqrt(2.0 * rest_p * (1.0 - rest_p))

    if number_of_classes == 2:
        numerator = np.sqrt(shots) * (0.5 - rest_p)
    elif number_of_classes == 3:
        other_probs = np.setdiff1d(probs[other_index], rest_p)
        numerator = np.sqrt(shots) * ((1. + np.abs(other_probs[0] - other_probs[1])) /
                                      number_of_classes - rest_p)
    else:
        raise ValueError('Do not support the number of class larger than three.')
    x = numerator / denominator
    loss_other = (1.) / (1. + np.exp(-x))
    loss[zero_index] = 0.0
    loss[one_index] = 1.0
    loss[other_index] = loss_other
    loss = np.mean(loss)
    return loss


def return_probabilities(counts, num_classes):
    """Return the probabilities of given measured counts
    Args:
        counts ([dict]): N data and each with a dict recording the counts
        num_classes (int): number of classes
    Returns:
        numpy.ndarray: NxK array
    """

    probs = np.zeros(((len(counts), num_classes)))
    for idx in range(len(counts)):
        count = counts[idx]
        shots = sum(count.values())
        for k, v in count.items():
            label = assign_label(k, num_classes)
            probs[idx][label] += v / shots
    return probs