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

"""
Unit test of Extrapolators based on a dictionary of point,
parameter pairs.
"""

import unittest
from sklearn import linear_model
from qiskit.chemistry.algorithms.pes_samplers.extrapolator import Extrapolator, \
    WindowExtrapolator, PolynomialExtrapolator, DifferentialExtrapolator, \
    PCAExtrapolator, SieveExtrapolator

from qiskit.aqua import AquaError

PARAM_DICT = {
    0.5: [0.07649726233077458, 1.2340960400591198e-07, 2.719308771599091e-08,
          0.008390437810203526, 0.0764741614750971, 1.9132000956096602e-07,
          2.9065333930913485e-07, 0.00833313170973392, -0.05158805745745537,
          5.8737804595134226e-08],
    0.6: [0.047537653925701376, -1.917048657008852e-07, -3.422658080883276e-08,
          0.004906396230989982, 0.0475356511244039, -1.4858989951517077e-07,
          -4.554118927702692e-07, 0.004904674017550646, -0.026469990208471097,
          -7.247195166355873e-08],
    0.7: [0.03613992230685178, 4.741673241363808e-07, 2.3773620079947958e-07,
          0.0019807112069739983, 0.03613265701497832, 6.110370616271814e-07,
          4.7624119913927746e-07, 0.0019793617137878546, -0.022690629295970738,
          1.4536323300165836e-07],
    0.8: [0.032353493452967126, -5.988966845753558e-07, 5.745328822729277e-08,
          0.00021194523430201692, 0.032355414067993846, -5.189590826114998e-07,
          -3.67836416341141e-07, 0.0002135481005546501, -0.021642050861260625,
          -1.7031569870078584e-08],
    0.9: [0.029140898456611407, -6.173482350529143e-07, 6.195627362572485e-08,
          -0.0003490974905228344, 0.029145442226752687, -6.121480980106215e-07,
          -7.301901035010772e-07, -0.0003490856477090481, -0.02152841574436387,
          1.7275126260813324e-07],
    1.0: [0.029247914836804657, 5.406923000869481e-08, -1.2835940341198057e-08,
          -0.00014550115747168215, 0.02924483275168305, 3.332715515541604e-08,
          -4.214594252866692e-08, -0.00014476700526947067, -0.022193103836975897,
          1.8066966314280466e-07],
    1.1: [0.03006212028509998, 8.719643359505152e-07, 2.2976675446724122e-07,
          0.00047712315923690516, 0.03006488639435572, 1.1166361133977898e-06,
          5.061212361236216e-07, 0.00047764287100316387, -0.023178549796925824,
          -2.928595563199974e-07],
    1.2: [0.030974376843233304, -1.0856144455895877e-06, -3.476503050548108e-07,
          0.001136538429249089, 0.03097485850614847, -9.341556711096642e-07,
          -1.1105021085720809e-07, 0.0011365812922166018, -0.024770225335378506,
          3.1946997465490094e-07],
    1.3: [0.031882221296263585, 1.786623717240475e-06, 5.966161740895298e-07,
          0.0019238138369525367, 0.03188025548265294, 2.001914958908424e-06,
          -7.558698586542756e-08, 0.0019267033837603463, -0.026633630436000855,
          -4.838673928102748e-07],
    1.4: [0.03363319046621523, -6.215327218045763e-06, -1.707461485292177e-06,
          0.0022111427295926026, 0.03363427344016048, -6.479433272163631e-06,
          -8.620279811840461e-07, 0.0022079369298442677, -0.029254200628083923,
          2.03258913595112e-06],
    1.5: [0.03566191437849682, 1.3175681716659443e-05, 3.3463916528882136e-06,
          0.0030670576873521546, 0.03565986755932079, 1.3808936313520536e-05,
          2.1227354591337757e-06, 0.0030639663487480417, -0.03203113690256062,
          -2.988438361808215e-06],
    1.6: [0.03853048160610931, -4.500510577352305e-05, -1.1042391095055013e-05,
          0.003589496950963951, 0.03852649109560952, -4.632560074669591e-05,
          -6.9604927841086826e-06, 0.003591766338853773, -0.03617535567521557,
          1.03526517642164e-05],
    1.7: [0.04166595503111059, 0.00012474608362087326, 3.0811181106852395e-05,
          0.004449408009656353, 0.04167583498336048, 0.0001291807564363206,
          1.9103762924011895e-05, 0.004443558543591776, -0.0411176424372442,
          -3.143959686889569e-05],
    1.8: [0.04630023768704881, -0.0003032527231323504, -7.224290210451026e-05,
          0.004988381942930891, 0.04629620402315099, -0.0003111138155773558,
          -4.900370932911525e-05, 0.004995942389375613, -0.047398106863887825,
          7.734110549927737e-05],
    1.9: [0.05237961421167222, 0.0006396923182584415, 0.00014873747649097767,
          0.005855974769304974, 0.05234227038906301, 0.0006540391246003456,
          0.00010652381338578109, 0.005850757199904456, -0.055346836396118364,
          -0.00018559571977688104]
}


class TestExtrapolators(unittest.TestCase):
    """Test Extrapolators."""

    def test_factory(self):
        """
        Test factory method implementation to create instances of various Extrapolators.
        """
        self.assertIsInstance(Extrapolator.factory(mode='window'), WindowExtrapolator)
        self.assertIsInstance(Extrapolator.factory(mode='poly'), PolynomialExtrapolator)
        self.assertIsInstance(Extrapolator.factory(mode='diff_model'), DifferentialExtrapolator)
        self.assertIsInstance(Extrapolator.factory(mode='pca'), PCAExtrapolator)
        self.assertIsInstance(Extrapolator.factory(mode='l1'), SieveExtrapolator)
        self.assertRaises(AquaError, Extrapolator.factory, mode="unknown")

    def test_polynomial_extrapolator(self):
        """
        Test extrapolation using a polynomial extrapolator with degree = 1 using all previous points
        in the parameters for extrapolation. This test confirms that the extrapolation of the
        parameters has a specified error relative to the actual parameter values.
        NOTE: The polynomial fit may give a runtime warning if the data is poorly fitted.
        This depends on degree and dataset and may need be tuned by the user to achieve
        optimal results. This reasoning holds for any instance using an internal
        polynomial extrapolator.
        """
        points = 0.7
        params = PolynomialExtrapolator(degree=3).extrapolate(points=[points],
                                                              param_dict=PARAM_DICT)
        sq_diff = [(actual - expected) ** 2 for actual, expected in
                   zip(params[points], PARAM_DICT[points])]
        self.assertLess(sum(sq_diff), 1e-3)

    def test_poly_window_extrapolator(self):
        """
        Test extrapolation using an WindowExtrapolator using a data window/lookback of 3 points
        and an internal polynomial extrapolator with degree = 1. This test confirms that no
        extrapolation is performed on points before the data window, i.e, the first two points,
        and that the extrapolation of the parameters on the last three points has a error below
        a threshold when compared to the actual parameter values.
        """
        points_interspersed = [.3, .5, .7, .8, 1.5]
        window_extrapolator = Extrapolator.factory("window",
                                                   extrapolator=PolynomialExtrapolator(degree=1),
                                                   window=3)
        params = window_extrapolator.extrapolate(points=points_interspersed, param_dict=PARAM_DICT)
        self.assertFalse(params.get(.3))
        self.assertFalse(params.get(.5))
        sq_diff_1 = [(actual - expected) ** 2
                     for actual, expected in zip(params[.7], PARAM_DICT[.7])]
        self.assertLess(sum(sq_diff_1), 1e-1)
        sq_diff_2 = [(actual - expected) ** 2
                     for actual, expected in zip(params[.8], PARAM_DICT[.8])]
        self.assertLess(sum(sq_diff_2), 1e-2)
        sq_diff_3 = [(actual - expected) ** 2
                     for actual, expected in zip(params[1.5], PARAM_DICT[1.5])]
        self.assertLess(sum(sq_diff_3), 1e-2)

    def test_differential_model_window_extrapolator(self):
        """
        Test extrapolation using an WindowExtrapolator using a data window/lookback of 3 points
        and an internal differential extrapolator with degree = 1 and the default linear regression
        model from scikit-learn. This test confirms that no extrapolation is performed on points
        before the data window, i.e, the first two points, and that the extrapolation of the
        parameters on the last three points has some specified error relative to the actual values.
        """
        points_interspersed = [.3, .5, .7, .8, 1.5]
        window_extrapolator = WindowExtrapolator(extrapolator=DifferentialExtrapolator(degree=1),
                                                 window=3)
        params = window_extrapolator.extrapolate(points=points_interspersed, param_dict=PARAM_DICT)
        self.assertFalse(params.get(.3))
        self.assertFalse(params.get(.5))
        sq_diff_1 = [(actual - expected) ** 2
                     for actual, expected in zip(params[.7], PARAM_DICT[.7])]
        self.assertLess(sum(sq_diff_1), 1e-2)
        sq_diff_2 = [(actual - expected) ** 2
                     for actual, expected in zip(params[.8], PARAM_DICT[.8])]
        self.assertLess(sum(sq_diff_2), 1e-3)
        sq_diff_3 = [(actual - expected) ** 2
                     for actual, expected in zip(params[1.5], PARAM_DICT[1.5])]
        self.assertLess(sum(sq_diff_3), 1e-3)

    def test_differential_model_window_alternate_model_extrapolator(self):
        """
        Test extrapolation using an WindowExtrapolator using a data window/lookback of 3 points
        and an internal differential extrapolator with degree = 1 and the Ridge regression model
        from scikit-learn. This test confirms that no extrapolation is performed on points before
        the data window, i.e, the first two points, and that the extrapolation of the parameters on
        the last three points has some specified error relative to the actual parameter values.
        """
        points_interspersed = [.3, .5, .7, .8, 1.5]
        model = linear_model.Ridge()
        window_extrapolator = WindowExtrapolator(extrapolator=DifferentialExtrapolator(degree=1,
                                                                                       model=model),
                                                 window=3)
        params = window_extrapolator.extrapolate(points=points_interspersed, param_dict=PARAM_DICT)
        self.assertFalse(params.get(.3))
        self.assertFalse(params.get(.5))
        sq_diff_1 = [(actual - expected) ** 2
                     for actual, expected in zip(params[.7], PARAM_DICT[.7])]
        self.assertLess(sum(sq_diff_1), 1e-2)
        sq_diff_2 = [(actual - expected) ** 2
                     for actual, expected in zip(params[.8], PARAM_DICT[.8])]
        self.assertLess(sum(sq_diff_2), 1e-3)
        sq_diff_3 = [(actual - expected) ** 2
                     for actual, expected in
                     zip(params[1.5], PARAM_DICT[1.5])]
        self.assertLess(sum(sq_diff_3), 1e-3)

    def test_pca_polynomial_window_extrapolator(self):
        """
        Test extrapolation using an PCAExtrapolator using a data window/lookback of 3 points
        and an internal polynomial extrapolator with degree = 1 using regular PCA as default.
        This test confirms that no extrapolation is performed on points before the
        data window, i.e, the first two points, and that the extrapolation of the parameters on
        last three points has a specified error relative to the actual parameter values.
        """
        points_interspersed = [.3, .5, .7, .8, 1.5]
        pca_poly_win_ext = PCAExtrapolator(extrapolator=PolynomialExtrapolator(degree=1),
                                           window=3)
        params = pca_poly_win_ext.extrapolate(points=points_interspersed, param_dict=PARAM_DICT)
        self.assertFalse(params.get(.3))
        self.assertFalse(params.get(.5))
        sq_diff_1 = [(actual - expected) ** 2
                     for actual, expected in zip(params[.7], PARAM_DICT[.7])]
        self.assertLess(sum(sq_diff_1), 1e-2)
        sq_diff_2 = [(actual - expected) ** 2
                     for actual, expected in zip(params[.8], PARAM_DICT[.8])]
        self.assertLess(sum(sq_diff_2), 1e-2)
        sq_diff_3 = [(actual - expected) ** 2
                     for actual, expected in zip(params[1.5], PARAM_DICT[1.5])]
        self.assertLess(sum(sq_diff_3), 1e-2)

    def test_sieve_poly_window_extrapolator(self):
        """
        Test extrapolation using an Sieve/Clustering Extrapolator using a data window/lookback of
        3 points and an internal polynomial extrapolator with degree = 1.
        This test confirms that no extrapolation is performed on points before the
        data window, i.e, the first two points, and that the extrapolation of the parameters on the
        last three points has some specified error relative to the actual parameter values.
        """
        points_interspersed = [.3, .5, .7, .8, 1.5]
        sieve_win_extrapolator = SieveExtrapolator(extrapolator=PolynomialExtrapolator(degree=1),
                                                   window=3)
        params = sieve_win_extrapolator.extrapolate(points=points_interspersed,
                                                    param_dict=PARAM_DICT)
        self.assertFalse(params.get(.3))
        self.assertFalse(params.get(.5))
        sq_diff_1 = [(actual - expected) ** 2
                     for actual, expected in zip(params[.7], PARAM_DICT[.7])]
        self.assertLess(sum(sq_diff_1), 1e-1)
        sq_diff_2 = [(actual - expected) ** 2
                     for actual, expected in zip(params[.8], PARAM_DICT[.8])]
        self.assertLess(sum(sq_diff_2), 1e-1)
        sq_diff_3 = [(actual - expected) ** 2
                     for actual, expected in zip(params[1.5], PARAM_DICT[1.5])]
        self.assertLess(sum(sq_diff_3), 1e-1)


if __name__ == '__main__':
    unittest.main()
