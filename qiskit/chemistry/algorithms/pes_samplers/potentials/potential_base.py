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


class EnergySurfaceBase(ABC):
    """ Class to hold a potential energy surface """

    def __init__(self, molecule):
        pass
    # self.update_molecule(molecule)

    # @abstractmethod
    # def update_molecule(self, molecule):
    #     """
    #     Wipe state if molecule changes, and check validity of molecule
    #     for potential.
    #     """

    @abstractmethod
    def eval(self, x):
        """
        After fitting the data to the fit function, predict the energy
            at a point x.
        """
        raise NotImplementedError

    @abstractmethod
    def fit_to_data(self, xdata, ydata, initial_vals=None, bounds_list=None):
        """ Fit the surface to data - x are coordinates, y is energy."""
        raise NotImplementedError

    @abstractmethod
    def get_equilibrium_geometry(self, scaling=1.0):
        """
        Returns the geometry for the minimal energy (scaled by 'scaling')
        Default units (scaling=1.0) are Angstroms. Scale by 1E-10 to get
        meters.
        """
        raise NotImplementedError

    @abstractmethod
    def get_minimal_energy(self, scaling=1.0):
        """
        Returns the value of the minimal energy (scaled by 'scaling')
        Default units (scaling=1.0) are J/mol. Scale appropriately for
        Hartrees.
        """
        raise NotImplementedError

    @abstractmethod
    def get_trust_region(self):
        """
        Returns the bounds of the region (in space) where the energy
        surface implementation can be trusted. When doing spline
        interpolation, for example, that would be the region where data
        is interpolated (vs. extrapolated) from the arguments of
        fit_to_data().
        """
        raise NotImplementedError


class VibronicStructureBase(ABC):
    """
    Class to hold a molecular vibronic structure providing access to
    vibrational modes and energy levels
    """

    def __init__(self, molecule):
        self.update_molecule(molecule)

    def update_molecule(self, molecule):
        """
        Wipe state if molecule changes, and check validity of molecule
        for potential.
        """
        self.molecule = molecule

    @abstractmethod
    def get_num_modes(self):
        """ returns the number of vibrational modes for the molecule """
        raise NotImplementedError

    @abstractmethod
    def vibrational_energy_level(self, n, mode=0):
        """ returns the n-th vibrational energy level for a given mode """
        raise NotImplementedError

    def get_maximum_trusted_level(self, mode=0):
        """
        Returns the maximum energy level for which the particular
        implementation still provides a good approximation of reality.
        Default value of 100. Redefined where needed (see e.g. Morse).
        """
        return 100

    '''
    # TODO: Do we need these? Does every vibrational mode have a fundamental
    #       frequency (e.g. when it is anharmonic)?

    @abstractmethod
    def fundamental_frequency(self, mode=0):
        """ returns the fundamental frequency for a given mode """
        raise NotImplementedError

    def wave_number(self, mode=0):
        """ returns the wave number for a given mode """
        return self.fundamental_frequency(mode) / C_CM_PER_S
    '''


class PotentialBase(EnergySurfaceBase, VibronicStructureBase):
    """
    Class to hold prescribed 1D potentials (e.g. Morse/Harmonic)
    over a degree of freedom.
    """

    def get_num_modes(self):
        """ This (1D) potential represents a single vibrational mode """
        return 1

    def get_trust_region(self):
        """
        The potential will usually be well-defined (even if not useful) for
        arbitrary x so we return a fairly large interval here.
        Redefine in derived classes if needed.
        """
        return (-100, 100)

    @abstractmethod
    def dissociation_energy(self, scaling=1.0):
        """ returns the dissociation energy (scaled by 'scaling')"""
        raise NotImplementedError


'''
class Potential1D(PotentialBase):
    def __init__(self, molecule, energy_surface, vibronic_struct):
        self.energy_surface = energy_surface
        self.vibronic_structure = vibronic_struct
        self.update_molecule(molecule)

    def update_molecule(self, molecule):
        """
        Wipe state if molecule changes, and check validity of molecule
        for potential.
        """
        self.molecule = molecule
        self.energy_surface.update_molecule(molecule)
        self.vibronic_structure.update_molecule(molecule)

    def eval(self, x):
        """
        After fitting the data to the fit function, predict the energy
            at a point x.
        """
        return self.energy_surface.eval(x)

    def fit_to_data(self, xdata, ydata, initial_vals=None, bounds_list=None):
        """ Fit the surface to data - x are coordinates, y is energy."""
        return self.energy_surface.fit_to_data(xdata, ydata, initial_vals,
                                               bounds_list)

    def get_equilibrium_geometry(self, scaling=1.0):
        """
        Returns the geometry for the minimal energy (scaled by 'scaling')
        Default units (scaling=1.0) are Angstroms. Scale by 1E-10 to get
        meters.
        """
        return self.energy_surface.get_equilibrium_geometry(scaling)

    def get_minimal_energy(self, scaling=1.0):
        """
        Returns the value of the minimal energy (scaled by 'scaling')
        Default units (scaling=1.0) are J/mol. Scale appropriately for
        Hartrees.
        """
        return self.energy_surface.get_minimal_energy(scaling)

    def get_num_modes(self):
        """ returns the number of vibrational modes for the molecule """
        return self.vibronic_structure.get_num_modes()

    def fundamental_frequency(self, mode=0):
        """ returns the fundamental frequency for a given mode """
        return self.vibronic_structure.fundamental_frequency(mode)

    def wave_number(self, mode=0):
        """ returns the wave number for a given mode """
        return self.fundamental_frequency(mode) / C_CM_PER_S

    def vibrational_energy_level(self, n, mode=0):
        """ returns the n-th vibrational energy level for a given mode """
        return self.vibronic_structure.vibrational_energy_level(n, mode)

    def get_maximum_trusted_level(self, mode = 0):
        """ returns the dissociation energy (scaled by 'scaling')"""
        return self.vibronic_structure.get_maximum_trusted_level(0)

    def dissociation_energy(self, scaling=1.0):
        """ returns the dissociation energy (scaled by 'scaling')"""
        return (self.eval(5) - self.get_minimal_energy() -
                self.vibrational_energy_level(0)) * scaling
'''
