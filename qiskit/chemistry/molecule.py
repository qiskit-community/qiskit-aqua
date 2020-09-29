# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2020
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or aexxont http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
#
"""
This module implements an interface for a generic molecule.
It defines the composing atoms (with properties like masses, and nuclear spin),
and allows for changing the molecular geometry through given degrees of freedom
(e.g. bond-stretching, angle-bending, etc.).
"""
import copy

import numpy as np
import scipy.linalg
from qiskit.chemistry.drivers import PySCFDriver, UnitsType, HFMethodType


class Molecule:
    """
    Molecule class
    """

    def __init__(self,
                 geometry,
                 degrees_of_freedom=None,
                 masses=None,
                 spins=None,
                 basis_set='sto3g',
                 hf_method=HFMethodType.RHF,
                 ):
        """
        Constructor.

        Args:
            geometry ([str, [float]]): 2d list containing atom string names
                to generate PySCF molecule strings as keys and list of 3
                floats representing Cartesian coordinates as values,
                in units of **Angstroms**.

            degrees_of_freedom ([callable]): List of functions taking a
                perturbation value and geometry and returns a perturbed
                geometry. Helper functions for typical perturbations are
                provided and can be used by the form
                itertools.partial(Molecule.stretching_potential,
                                      {'atom_pair': (1, 2))
                to specify the desired degree of freedom.

            masses([float]): The list of masses of each atom.
                If provided, must be the same length as number of atoms
                in geometry.
        """
        self._geometry = geometry
        self._degrees_of_freedom = degrees_of_freedom

        if masses is not None and not len(masses) == len(self._geometry):
            raise ValueError(
                'Length of masses must match length of geometries, '
                'found {} and {} respectively'.format(
                    len(masses),
                    len(self._geometry)
                )
            )

        self._masses = masses

        if spins is not None and not len(spins) == len(self._geometry):
            raise ValueError(
                'Length of spins must match length of geometries, '
                'found {} and {} respectively'.format(
                    len(spins),
                    len(self._geometry)
                )
            )

        self._spins = spins
        self._basis_set = basis_set
        self._hf_method = hf_method

    @classmethod
    def __distance_modifier(cls, function, parameter, geometry, atom_pair):
        """
        Args:
            atom_pair(tuple(int)): A tuple with two integers, indexing
                which atoms from the starting geometry should be moved
                apart. **Atom1 is moved away from Atom2, while Atom2
                remains stationary.**
            function: a function of two parameters (current distance,
                extra parameter) returning the new distance
            parameter(float): The extra parameter of the function above.
            geometry(([str, [float]])): The initial geometry to perturb.
        """
        a1, a2 = atom_pair
        startingCoord1 = np.array(geometry[a1][1])
        coord2 = np.array(geometry[a2][1])

        startingDistanceVector = startingCoord1 - coord2
        startingL2distance = np.linalg.norm(startingDistanceVector)
        newL2distance = function(startingL2distance, parameter)
        newDistanceVector = startingDistanceVector * (
            newL2distance / startingL2distance
        )
        newCoord1 = coord2 + newDistanceVector

        ending_geometry = copy.deepcopy(geometry)
        ending_geometry[a1][1] = newCoord1.tolist()
        return ending_geometry

    @classmethod
    def absolute_distance(cls, distance, geometry, atom_pair):
        """
        Args:
            atom_pair(tuple(int)): A tuple with two integers,
                indexing which atoms from the starting geometry should be
                moved apart. **Atom1 is moved away (at the given distance)
                from Atom2, while Atom2 remains stationary.**
            distance(float): The (new) distance between the two atoms.
            geometry(([str, [float]])): The initial geometry to perturb.
        """

        def func(x, d): return d

        return cls.__distance_modifier(func, distance, geometry, atom_pair)

    @classmethod
    def absolute_stretching(cls, perturbation, geometry, atom_pair):
        """
        Args:
            atom_pair(tuple(int)): A tuple with two integers,
                indexing which atoms from the starting geometry should be
                stretched apart. **Atom1 is stretched away from Atom2, while
                Atom2 remains stationary.**
            perturbation(float): The magnitude of the stretch.
                (New distance = stretch + old distance)
            geometry(([str, [float]])): The initial geometry to perturb.
        """

        def func(x, d): return x + d

        return cls.__distance_modifier(func, perturbation, geometry,
                                       atom_pair)

    @classmethod
    def relative_stretching(cls, perturbation, geometry, atom_pair):
        """
        Args:
            atom_pair(tuple(int)): A tuple with two integers, indexing which
                atoms from the starting geometry should be stretched apart.
                **Atom1 is stretched away from Atom2, while Atom2 remains
                stationary.**
            perturbation(float): The magnitude of the stretch.
                (New distance = stretch * old distance)
            geometry(([str, [float]])): The initial geometry to perturb.
        """

        def func(x, d): return x * d

        return cls.__distance_modifier(func, perturbation, geometry,
                                       atom_pair)

    @classmethod
    def __bend_modifier(cls, function, parameter, geometry, atom_trio):
        """
        Args:
            atom_trio(tuple(int)): A tuple with three integers, indexing
                which atoms from the starting geometry should be bent apart.
                **Atom1 is bent *away* from Atom3 by an angle whose vertex
                is Atom2, while Atom2 and Atom3 remain stationary.**
            function: a function of two parameters (current angle,
                extra parameter) returning the new angle
            parameter(float): The extra parameter of the function above.
            geometry(([str, [float]])): The initial geometry to perturb.
        """
        a1, a2, a3 = atom_trio
        startingCoord1 = np.array(geometry[a1][1])
        coord2 = np.array(geometry[a2][1])
        coord3 = np.array(geometry[a3][1])

        distanceVec1to2 = startingCoord1 - coord2
        distanceVec3to2 = coord3 - coord2
        rot_axis = np.cross(distanceVec1to2, distanceVec3to2)
        # If atoms are linear, choose the rotation direction randomly,
        # but still along the correct plane
        # Maybe this is a bad idea if we end up here on some
        # existing bending path.
        # It'd be good to fix this later to remember the axis in some way.
        if np.linalg.norm(rot_axis) == 0:
            nudged_vec = copy.deepcopy(distanceVec1to2)
            nudged_vec[0] += .01
            rot_axis = np.cross(nudged_vec, distanceVec3to2)
        rot_unit_axis = rot_axis / np.linalg.norm(rot_axis)
        startingAngle = np.arcsin(
            np.linalg.norm(rot_axis) / (
                np.linalg.norm(distanceVec1to2)
                * np.linalg.norm(distanceVec3to2)
            )
        )
        newAngle = function(startingAngle, parameter)
        perturbation = newAngle - startingAngle
        rot_matrix = scipy.linalg.expm(
            np.cross(
                np.eye(3),
                rot_unit_axis *
                perturbation))
        newCoord1 = rot_matrix @ startingCoord1

        ending_geometry = copy.deepcopy(geometry)
        ending_geometry[a1][1] = newCoord1.tolist()
        return ending_geometry

    @classmethod
    def absolute_angle(cls, angle, geometry, atom_trio):
        """
        Args:
            atom_trio(tuple(int)): A tuple with three integers, indexing
                which atoms from the starting geometry should be bent apart.
                **Atom1 is bent *away* from Atom3 by an angle whose vertex
                is Atom2 and equal to **angle**, while Atom2 and Atom3
                remain stationary.**
            angle(float): The magnitude of the perturbation in **radians**.
                **Positive bend is always in the direction toward Atom3.**
                the direction of increasing the starting angle.**
            geometry(([str, [float]])): The initial geometry to perturb.
        """

        def func(x, d): return d

        return cls.__bend_modifier(func, angle, geometry, atom_trio)

    @classmethod
    def absolute_bending(cls, bend, geometry, atom_trio):
        """
        Args:
            atom_trio(tuple(int)): A tuple with three integers, indexing
                which atoms from the starting geometry should be bent apart.
                **Atom1 is bent *away* from Atom3 by an angle whose vertex
                is Atom2 and equal to the initial angle **plus** bend,
                while Atom2 and Atom3 remain stationary.**
            bend(float): The magnitude of the perturbation in **radians**.
                **Positive bend is always in the direction toward Atom3.**
                the direction of increasing the starting angle.**
            geometry(([str, [float]])): The initial geometry to perturb.
        """

        def func(x, d): return x + d

        return cls.__bend_modifier(func, bend, geometry, atom_trio)

    @classmethod
    def relative_bending(cls, bend, geometry, atom_trio):
        """
        Args:
            atom_trio(tuple(int)): A tuple with three integers,
                indexing which atoms from the starting geometry
                should be bent apart. **Atom1 is bent *away* from Atom3
                by an angle whose vertex is Atom2 and equal to the initial
                angle **times** bend, while Atom2 and Atom3
                remain stationary.**
            bend(float): The magnitude of the perturbation in **radians**.
                **Positive bend is always in the direction toward Atom3.**
            the direction of increasing the starting angle.**
            geometry(([str, [float]])): The initial geometry to perturb.
        """

        def func(x, d): return x * d

        return cls.__bend_modifier(func, bend, geometry, atom_trio)

    def get_perturbed_geom(self, perturbations=None):
        if not perturbations or not self._degrees_of_freedom:
            return self._geometry
        geometry = copy.deepcopy(self._geometry)
        for per, dof in zip(perturbations, self._degrees_of_freedom):
            geometry = dof(per, geometry)
        return geometry

    @classmethod
    def get_geometry_str(cls, geometry):
        return '; '.join([name + ' ' + ', '.join(map(str, coord))
                          for (name, coord) in geometry])

    @property
    def geometry_str(self):
        return get_geometry_str(geometry)

    @property
    def basis_set(self):
        return self._basis_set

    @property
    def hf_method(self):
        return self._hf_method

    @property
    def spins(self):
        return self._spins

    @property
    def masses(self):
        return self._masses

