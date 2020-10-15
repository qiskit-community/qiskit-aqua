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
This module implements a 1D Morse potential.
"""
import numpy as np
from scipy.optimize import curve_fit

from qiskit.chemistry.algorithms.pes_samplers.potentials.potential_base import PotentialBase
import qiskit.chemistry.constants as const


class MorsePotential(PotentialBase):
    """
    Implements a 1D Morse potential.
    Input units are Angstroms (distance between the two atoms),
        and output units are Hartrees (molecular energy).
    """
    # Works in Angstroms and Hartrees

    def __init__(self, molecule):
        """
        Initializes the potential to the zero-function.
        fit() should be used afterwards to fit the potential to
            computed molecular energies.

        Args:
            molecule: the underlying molecule.

        Raises:
            ValueError: Only implemented for diatomic molecules
        """
        # Initialize with zero-potential.
        # Later - fit energy values (fit)
        self.d_e = 0.0
        self.m_shift = 0.0
        self.alpha = 0.0
        self.r_0 = 0.0
        self._mA = molecule.masses[0]
        self._mB = molecule.masses[1]

    @staticmethod
    def fit_function(x, d_e, alpha, r_0, m_shift):
        """
        Functional form of the potential.
        """
        # d_e (Hartree), alpha (1/Angstrom), r_0 (Angstrom)
        return d_e * (1 - np.exp(-alpha * (x - r_0))) ** 2 + m_shift

    def eval(self, x):
        """
        Evaluates the potential at a given point.
        """
        # Expects Angstroms returns Hartrees
        return self.fit_function(x, self.d_e,
                                 self.alpha, self.r_0, self.m_shift)

    def update_molecule(self, molecule):
        """
        Updates the underlying molecule.

        Raises:
            ValueError: Only implemented for diatomic molecules
        """
        # Check the provided molecule
        if len(molecule.masses) != 2:
            raise ValueError(
                'Morse potential only works for diatomic molecules!')
        self._mA = molecule.masses[0]
        self._mB = molecule.masses[1]

    def fit(self, xdata, ydata, initial_vals=None, bounds_list=None):
        """
        Fits a potential to computed molecular energies.

        Args:
            xdata: interatomic distance points (Angstroms)
            ydata: molecular energies (Hartrees)
            initial_vals: Initial values for fit parameters. None for default.
                    Order of parameters is d_e, alpha, r_0 and m_shift
                    (see fit_function implementation)
            bounds_list: Bounds for the fit parameters. None for default.
                    Order of parameters is d_e, alpha, r_0 and m_shift
                    (see fit_function implementation)
        """

        # do the Morse potential fit
        # here, the order of parameters is
        # [d_e (Hartree), alpha (1/ang), r_0 (ang), energy_shift (Hartree)]
        m_p0 = (initial_vals if initial_vals is not None
                else np.array([0.25, 2, 0.735, 1.5]))
        m_bounds = (bounds_list if bounds_list is not None
                    else ([0, 0, 0.3, -5],
                          [2.5, np.inf, 1.0, 5]))

        fit, _ = curve_fit(self.fit_function, xdata, ydata,
                           p0=m_p0, maxfev=100000, bounds=m_bounds)

        self.d_e = fit[0]
        self.alpha = fit[1]
        self.r_0 = fit[2]
        self.m_shift = fit[3]

    def get_equilibrium_geometry(self, scaling=1.0):
        """
        Returns the interatomic distance corresponding to minimal energy.
        Args:
            scaling: Scaling to change units. (Default is 1.0 for Angstroms)
        """
        # TODO: Should this be moved to Molecule, given that the meaning of
        # the returned value(s) is defined by the molecule's degrees of
        # freedom?

        # Returns the distance for the minimal energy (scaled by 'scaling')
        # Default units (scaling=1.0) are Angstroms. Scale by 1E-10 to get
        # meters.
        return self.r_0 * scaling

    def get_minimal_energy(self, scaling=1.0):
        """
        Returns the smallest molecular energy for the current fit.
        Args:
            scaling: Scaling to change units. (Default is 1.0 for Hartrees)
        """
        # Returns the distance for the minimal energy (scaled by 'scaling'')
        # Default units (scaling=1.0) are Hartrees. Scale appropriately for
        # Joules (per mol).
        return self.m_shift * scaling

    def dissociation_energy(self, scaling=1.0):
        """
        Returns the calculated dissociation energy for the current fit.
        Args:
            scaling: Scaling to change units. (Default is 1.0 for Hartrees)
        """
        # Returns the dissociation energy (scaled by 'scaling').
        # Default units (scaling=1.0) are Hartrees. Scale appropriately for
        # Joules (per mol).
        de = self.d_e
        diss_nrg = de - self.vibrational_energy_level(0)

        return diss_nrg * scaling

    def fundamental_frequency(self):
        """
        Returns the fundamental frequency for the current fit (in s^-1).
        """
        de = self.d_e * const.HARTREE_TO_J  # Hartree, need J/molecule
        alp = self.alpha * 1E10  # 1/angstrom, need 1/meter
        # r0 = self.r_0*1E-10  # angstrom, need meter
        mr = (self._mA * self._mB) / (self._mA + self._mB)

        # omega_0 in units rad/s converted to 1/s by dividing by 2Pi
        omega_0 = (np.sqrt((2 * de * alp ** 2) / mr)) / (2 * np.pi)

        # fundamental frequency in s**-1
        return omega_0

    def wave_number(self):
        """
        Returns the wave number for the current fit (in cm^-1).
        """
        return self.fundamental_frequency() / const.C_CM_PER_S

    def vibrational_energy_level(self, n):
        """
        Returns the n-th vibrational energy level for the current fit
            (in Hartrees).
        """
        de = self.d_e * const.HARTREE_TO_J  # Hartree, need J/molecule

        omega_0 = self.fundamental_frequency()
        e_n = const.H_J_S * omega_0 * (n + 0.5) - \
            ((const.H_J_S * omega_0 * (n + 0.5)) ** 2) / (4 * de)

        # energy level
        return e_n * const.J_TO_HARTREE

    def get_maximum_trusted_level(self, mode=0):
        # For the formula below, see
        # "Partition Functions", by Popovas A., et.al
        # Astronomy & Astrophysics, Vol. 595, November 2016
        # https://doi.org/10.1051/0004-6361/201527209
        return np.floor(
            2 * self.dissociation_energy(const.HARTREE_TO_J) /
            (const.H_J_S * self.fundamental_frequency())
        )
