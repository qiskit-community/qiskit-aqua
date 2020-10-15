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
This module declares interfaces for implementing potential energy surface
and vibrational structure of a given molecule.
"""
from abc import ABC, abstractmethod
from typing import Tuple
from qiskit.chemistry.drivers import Molecule

class EnergySurfaceBase(ABC):
    """ Class to hold a potential energy surface """

    @abstractmethod
    def eval(self, x)-> float:
        """
        After fitting the data to the fit function, predict the energy
            at a point x.
        Args:
            x: value to evaluate surface in

        Returns:
            value of surface in point x
        """
        raise NotImplementedError

    @abstractmethod
    def fit(self, xdata, ydata, initial_vals=None, bounds_list=None)->None:
        """
        Fits surface to data
        Args:
            xdata: x data to be fitted
            ydata: y data to be fitted

        Returns:
            fitted surface
        """
        raise NotImplementedError

    @abstractmethod
    def get_equilibrium_geometry(self, scaling=1.0)->float:
        """
        Returns the geometry for the minimal energy (scaled by 'scaling')
        Default units (scaling=1.0) are Angstroms. Scale by 1E-10 to get
        meters.
        Args:
            scaling: scaling factor

        Returns:
            equilibrium geometry
        """
        raise NotImplementedError

    @abstractmethod
    def get_minimal_energy(self, scaling=1.0)->float:
        """
        Returns the value of the minimal energy (scaled by 'scaling')
        Default units (scaling=1.0) are J/mol. Scale appropriately for
        Hartrees.
        Args:
            scaling: scaling factor

        Returns:
            minimum energy
        """
        raise NotImplementedError

    @abstractmethod
    def get_trust_region(self)->Tuple[float,float]:
        """
        Returns the bounds of the region (in space) where the energy
        surface implementation can be trusted. When doing spline
        interpolation, for example, that would be the region where data
        is interpolated (vs. extrapolated) from the arguments of
        fit().

        Returns:
            the trust region between bounds
        """
        raise NotImplementedError


class VibronicStructureBase(ABC):
    """
    Class to hold a molecular vibronic structure providing access to
    vibrational modes and energy levels
    """

    def __init__(self, molecule):
        self.update_molecule(molecule)

    def update_molecule(self, molecule)->Molecule:
        """
        Wipe state if molecule changes, and check validity of molecule
        for potential.
        Args:
            molecule: chemistry molecule

        Returns:
            molecule used
        """
        self.molecule = molecule

    @abstractmethod
    def get_num_modes(self)-> float:
        """
        Returns the number of vibrational modes for the molecule
        Returns:
            the number of vibrational modes
        """
        raise NotImplementedError

    @abstractmethod
    def vibrational_energy_level(self, n, mode=0) -> float:
        """
        Returns the n-th vibrational energy level for a given mode
        Args:
            n: number of vibrational mode
            mode: vibrational mode

        Returns:
            n-th vibrational energy level for a given mode
        """
        raise NotImplementedError

    def get_maximum_trusted_level(self, mode=0)->float:
        """
        Returns the maximum energy level for which the particular
        implementation still provides a good approximation of reality.
        Default value of 100. Redefined where needed (see e.g. Morse).
        Args:
            mode: vibronic mode

        Returns:
            maximum_trusted_level setted
        """
        return 100


class PotentialBase(EnergySurfaceBase, VibronicStructureBase):
    """
    Class to hold prescribed 1D potentials (e.g. Morse/Harmonic)
    over a degree of freedom.
    """

    def get_num_modes(self)->int:
        """ This (1D) potential represents a single vibrational mode """
        return 1

    def get_trust_region(self)->Tuple[float,float]:
        """
        The potential will usually be well-defined (even if not useful) for
        arbitrary x so we return a fairly large interval here.
        Redefine in derived classes if needed.
        """
        return (-100, 100)

    @abstractmethod
    def dissociation_energy(self, scaling=1.0)->float:
        """ returns the dissociation energy (scaled by 'scaling')"""
        raise NotImplementedError
