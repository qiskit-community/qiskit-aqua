# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# Imports
import numpy as np
import unittest
# TODO: Change back to this before PR: from test.aqua.common import QiskitAquaTestCase
from common import QiskitAquaTestCase
from qiskit.aqua.algorithms.single_sample.recommendation_systems import QuantumRecommendation


class TestQuantumRecommendation(QiskitAquaTestCase):
    """Unit tests for QuantumRecommendation."""
    def test_threshold_to_control_string_four_bits(self):
        """Tests converting a threshold singular value to a control string for the thresholding circuit."""
        pref = np.identity(2)
        qrs = QuantumRecommendation(preference_matrix=pref, nprecision_bits=4)
        possible_sigmas = qrs._qsve.possible_estimated_singular_values(4)[:-1]

        # Correct values determined analytically by evaluating cos(pi * theta).
        # For example, for the first string "1000", theta = 1/2, so cos(pi / 2) = 0, as desired.
        correct_strings = ["1000", "0111", "0110", "0101", "0100", "0011", "0010", "0001", "0000"]

        for (ii, sigma) in enumerate(possible_sigmas):
            self.assertEqual(qrs._threshold_to_control_string(sigma), correct_strings[ii])

    def test_threshold_to_control_string_extreme_values(self):
        """Tests converting a threshold singular value to a control string for the thresholding circuit.

        This tests the edge case threshold = 0 and threshold = 1 for different numbers of precision bits.
        """
        for n in range(1, 10):
            pref = np.identity(2)
            qrs = QuantumRecommendation(preference_matrix=pref, nprecision_bits=n)
            self.assertEqual(qrs._threshold_to_control_string(0), "1" + "0" * (n - 1))
            self.assertEqual(qrs._threshold_to_control_string(1), "0" * n)

    def test_identity2by2_binary(self):
        """Tests that recommendations for an identity preference matrix with binary ratings are correct."""
        # Make a preference matrix and recommendation system
        pref = np.array([[1, 0],
                         [0, 1]])
        qrs = QuantumRecommendation(pref, nprecision_bits=3)

        # Test the quantum recommendations for user [1, 0]
        user = np.array([1, 0])
        products, probabilities = qrs.recommend(user, threshold=0.0)
        self.assertEqual(products, [0])
        self.assertAlmostEqual(probabilities, [1.0])

        # Test the quantum recommendations for user [0, 1]
        user = np.array([0, 1])
        products, probabilities = qrs.recommend(user, threshold=0.0)
        self.assertEqual(products, [1])
        self.assertAlmostEqual(probabilities, [1.0])

    def test_identity2by2_float(self):
        """Tests that recommendations for an identity preference matrix with floating point ratings are correct."""
        # Make a preference matrix and recommendation system
        pref = np.array([[1, 0],
                         [0, 1]])
        qrs = QuantumRecommendation(pref, nprecision_bits=3)

        # Test the quantum recommendations for a user
        user = np.array([0.6, 0.8])

        # Test with a threshold of 0 (full rank matrix)
        prods, probs = qrs.recommend(user, threshold=0.0)
        self.assertEqual(set(prods), {0, 1})
        self.assertTrue(np.allclose(list(sorted(probs)), [0.36, 0.64], atol=1e-2))

    def test4by4rank1(self):
        """Tests rank 1 preference matrix with four users and four products using different threshold values."""
        pref = np.array([[1, 1, 0, 0],
                         [1, 1, 0, 0],
                         [1, 1, 0, 0],
                         [1, 1, 0, 0]])
        qrs = QuantumRecommendation(preference_matrix=pref, nprecision_bits=3)
        user = np.array([1, 0, 0, 0])

        # Test with a threshold corresponding to rank 1
        products, probs = qrs.recommend(user, threshold=0.80)
        self.assertEqual(set(products), {0, 1})
        self.assertTrue(np.allclose(probs, [0.5, 0.5], atol=0.2))

        # Test with a threshold corresponding to full rank
        products, probs = qrs.recommend(user, threshold=0.00)
        self.assertEqual(products, [0])
        self.assertTrue(np.allclose(probs, [1], atol=1e-2))

    def test4by4rank2(self):
        """Tests a rank 2 preference matrix with four users and four products."""
        pref = np.array([[1, 1, 0, 0],
                         [0, 1, 1, 0],
                         [0, 1, 1, 0],
                         [1, 1, 0, 0]])
        qrs = QuantumRecommendation(preference_matrix=pref, nprecision_bits=3)
        user = np.array([1, 0, 0, 0])

        # Recommend with a threshold of zero (keep all singular values)
        prods, probs = qrs.recommend(user, threshold=0)
        self.assertEqual(prods, [0])
        self.assertEqual(probs, [1.0])

        # Recommend with a higher threshold to keep less singular values
        prods, _ = qrs.recommend(user, threshold=0.9)
        cprods, _ = qrs.classical_recommendation(user, rank=1)
        self.assertEqual(set(prods), set(cprods))

    def test4by4rank2_high_precision(self):
        """Tests that the output probability distributions for classical/quantum are close with 5 precision qubits."""
        pref = np.array([[1, 1, 0, 0],
                         [1, 1, 0, 0],
                         [1, 0, 0, 1],
                         [1, 0, 0, 1]])
        qrs = QuantumRecommendation(preference_matrix=pref, nprecision_bits=5)
        user = np.array([1, 0, 0, 0])

        # Recommend with a threshold of zero (keep all singular values)
        prods, probs = qrs.recommend(user, threshold=0)
        self.assertEqual(prods, [0])
        self.assertEqual(probs, [1.0])

        # Recommend with a higher threshold to keep less singular values
        prods, probs = qrs.recommend(user, threshold=0.9)
        probs = list(sorted(probs))
        cprods, cprobs = qrs.classical_recommendation(user, rank=1)
        cprobs = list(sorted(cprobs))
        self.assertEqual(set(prods), set(cprods))
        self.assertTrue(np.allclose(probs, cprobs, atol=0.1))


if __name__ == "__main__":
    unittest.main()
