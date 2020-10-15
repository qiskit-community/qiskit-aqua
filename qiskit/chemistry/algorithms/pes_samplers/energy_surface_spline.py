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
An spline interpolation method for data fitting
"""

import scipy.interpolate as interp
from scipy.optimize import minimize_scalar

from qiskit.chemistry.algorithms.pes_samplers.potentials.potential_base import EnergySurfaceBase


class EnergySurface1DSpline(EnergySurfaceBase):
    """
    A simple cubic spline interpolation for the potential energy surface.
    """

    def __init__(self):
        """
        Constructor.
        """
        super().__init__()
        self._eval = None
        self.eval_d = None
        self.min_x = None
        self.min_val = None
        self.x_left = None
        self.x_right = None

    # Implementing the EnergySurfaceBase interface

    def eval(self, x):
        """
        After fitting the data to the fit function, predict the energy
            at a point x.
        """
        assert self._eval is not None
        result = self._eval(x)

        return result

    def fit(self, xdata, ydata, initial_vals=None, bounds_list=None):

        newx = xdata
        newy = ydata

        tck = interp.splrep(newx, newy, k=3)

        self._eval = lambda x: interp.splev(x, tck)
        self.eval_d = lambda x: interp.splev(x, tck, der=1)

        result = minimize_scalar(self._eval)
        assert result.success

        self.min_x = result.x
        self.min_val = result.fun
        self.x_left = min(xdata)
        self.x_right = max(xdata)

    def get_equilibrium_geometry(self, scaling=1.0):
        """
        Returns the geometry for the minimal energy (scaled by 'scaling')
        Default units (scaling=1.0) are Angstroms. Scale by 1E-10 to get
        meters.
        """
        assert self.min_x is not None
        return self.min_x * scaling

    def get_minimal_energy(self, scaling=1.0):
        """
        Returns the value of the minimal energy (scaled by 'scaling')
        Default units (scaling=1.0) are J/mol. Scale appropriately for
        Hartrees.
        """
        assert self.min_val is not None
        return self.min_val * scaling

    def get_trust_region(self):
        """
        Returns the bounds of the region (in space) where the energy
        surface implementation can be trusted. When doing spline
        interpolation, for example, that would be the region where data
        is interpolated (vs. extrapolated) from the arguments of
        fit().
        """
        return (self.x_left, self.x_right)
