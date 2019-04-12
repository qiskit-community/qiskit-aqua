import numpy as np
import unittest
from test.common import QiskitAquaTestCase
from qiskit.aqua.utils import pdf_w, pdf_a, circ_dist, angle_to_value


class TestCircDist(QiskitAquaTestCase):
    def test_circ_dist(self):
        # Scalars
        self.assertEqual(circ_dist(0, 0), 0)
        self.assertAlmostEqual(circ_dist(0, 0.5), 0.5)
        self.assertAlmostEqual(circ_dist(0.2, 0.7), 0.5)
        self.assertAlmostEqual(circ_dist(0.1, 0.9), 0.2)
        self.assertAlmostEqual(circ_dist(0.9, 0.1), 0.2)

        # Arrays
        w0 = np.array([0, 0.2, 1])
        w1 = 0.2
        expected = np.array([0.2, 0, 0.2])
        actual = circ_dist(w0, w1)
        actual_swapped = circ_dist(w1, w0)

        for e, a, s in zip(expected, actual, actual_swapped):
            self.assertAlmostEqual(e, a)
            self.assertAlmostEqual(e, s)


class TestPDFs(QiskitAquaTestCase):

    def test_pdf_w(self):
        m = [1, 2, 3, 10, 100]
        w_exact = [0,   0.2, 0.2, 0.5, 0.8]
        w = [0.1, 0.2, 0.9, 1.0, 0.79999999]
        w_expected = [0.9045084972,
                      1,
                      0.0215932189,
                      0,
                      0]

        for mi, wi_exact, wi, wi_expected in zip(m, w_exact, w, w_expected):
            self.assertAlmostEqual(wi_expected, pdf_w(wi, wi_exact, mi))

    def test_pdf_a(self):
        m = [1, 2, 3, 10, 100]
        a_exact = np.array([0,   0.2, 0.2, 0.5, 0.8])
        a = angle_to_value([0, 3 / 4, 1 / 8, 250 / 1024, 0.79999999])
        a_expected = [1,
                      0.6399999999999995,
                      0.9065420129264011,
                      0,
                      0]

        for mi, ai_exact, ai, ai_expected in zip(m, a_exact, a, a_expected):
            self.assertAlmostEqual(ai_expected, pdf_a(ai, ai_exact, mi))


if __name__ == "__main__":
    unittest.main()
