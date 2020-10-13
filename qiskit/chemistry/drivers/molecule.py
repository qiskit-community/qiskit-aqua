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

"""Driver-independent Molecule definition."""

from typing import Callable, Tuple, List, Optional
import copy

import numpy as np
import scipy.linalg

from .units_type import UnitsType


class Molecule:
    """Driver-independent Molecule definition.

    This module implements an interface for a driver-independent, i.e. generic molecule
    definition. It defines the composing atoms (with properties like masses),
    and allows for changing the molecular geometry through given degrees of freedom
    (e.g. bond-stretching, angle-bending, etc.). The geometry as provided in the
    constructor can be affected, through setting perturbations, and it is this perturbed
    geometry that is supplied by the geometry getter. Setting perturbations to None will
    cause the original geometry to be returned, and there is a getter to get this value
    directly if its needed.
    """

    def __init__(self,
                 geometry: List[Tuple[str, List[float]]],
                 multiplicity: int = 1,
                 charge: int = 0,
                 degrees_of_freedom: Optional[List[Callable]] = None,
                 masses: Optional[List[float]] = None
                 ) -> None:
        """
        Args:
            geometry: A list of atoms defining a given molecule where each item in the list
                is an atom name together with a list of 3 floats representing the x,y and z
                Cartesian coordinates of the atom's position in units of **Angstrom**.
            multiplicity: Multiplicity (2S+1) of the molecule
            charge: Charge on the molecule
            degrees_of_freedom: List of functions taking a
                perturbation value and geometry and returns a perturbed
                geometry. Helper functions for typical perturbations are
                provided and can be used by the form
                itertools.partial(Molecule.stretching_potential,{'atom_pair': (1, 2))
                to specify the desired degree of freedom.
            masses: Mass of each atom the molecule may optionally be provided.

        Raises:
            ValueError: Length of masses must match length of geometries.
        """
        Molecule._check_consistency(geometry, masses)

        self._geometry = geometry
        self._degrees_of_freedom = degrees_of_freedom
        self._multiplicity = multiplicity
        self._charge = charge
        self._masses = masses

        self._perturbations = None  # type: Optional[List[float]]

    @staticmethod
    def _check_consistency(geometry: List[Tuple[str, List[float]]],
                           masses: Optional[List[float]]):
        if masses is not None and len(masses) != len(geometry):
            raise ValueError('Length of masses {} must match length of geometries {}'.format(
                len(masses), len(geometry)))

    @classmethod
    def _distance_modifier(cls,
                           function: Callable[[float, float], float],
                           parameter: float,
                           geometry: List[Tuple[str, List[float]]],
                           atom_pair: Tuple[int, int]) -> List[Tuple[str, List[float]]]:
        """
        Args:
            function: a function of two parameters (current distance,
                extra parameter) returning the new distance
            parameter: The extra parameter of the function above.
            geometry: The initial geometry to perturb.
            atom_pair: A tuple with two integers, indexing
                which atoms from the starting geometry should be moved
                apart. **Atom1 is moved away from Atom2, while Atom2
                remains stationary.**

        Returns:
            end geometry
        """
        a_1, a_2 = atom_pair
        starting_coord1 = np.array(geometry[a_1][1])
        coord2 = np.array(geometry[a_2][1])

        starting_distance_vector = starting_coord1 - coord2
        starting_l2distance = np.linalg.norm(starting_distance_vector)
        new_l2distance = function(starting_l2distance, parameter)
        new_distance_vector = starting_distance_vector * (
            new_l2distance / starting_l2distance
        )
        new_coord1 = coord2 + new_distance_vector

        ending_geometry = copy.deepcopy(geometry)
        ending_geometry[a_1] = ending_geometry[a_1][0], new_coord1.tolist()
        return ending_geometry

    @classmethod
    def absolute_distance(cls,
                          distance: float,
                          geometry: List[Tuple[str, List[float]]],
                          atom_pair: Tuple[int, int]) -> List[Tuple[str, List[float]]]:
        """
        Args:
            distance: The (new) distance between the two atoms.
            geometry: The initial geometry to perturb.
            atom_pair: A tuple with two integers,
                indexing which atoms from the starting geometry should be
                moved apart. **Atom1 is moved away (at the given distance)
                from Atom2, while Atom2 remains stationary.**

        Returns:
            end geometry
        """

        def func(curr_dist, extra):  # pylint: disable=unused-argument
            return extra

        return cls._distance_modifier(func, distance, geometry, atom_pair)

    @classmethod
    def absolute_stretching(cls,
                            perturbation: float,
                            geometry: List[Tuple[str, List[float]]],
                            atom_pair: Tuple[int, int]) -> List[Tuple[str, List[float]]]:
        """
        Args:
            perturbation: The magnitude of the stretch.
                (New distance = stretch + old distance)
            geometry: The initial geometry to perturb.
            atom_pair: A tuple with two integers,
                indexing which atoms from the starting geometry should be
                stretched apart. **Atom1 is stretched away from Atom2, while
                Atom2 remains stationary.**

        Returns:
            end geometry
        """

        def func(curr_dist, extra):
            return curr_dist + extra

        return cls._distance_modifier(func, perturbation, geometry,
                                      atom_pair)

    @classmethod
    def relative_stretching(cls,
                            perturbation: float,
                            geometry: List[Tuple[str, List[float]]],
                            atom_pair: Tuple[int, int]) -> List[Tuple[str, List[float]]]:
        """
        Args:
            perturbation: The magnitude of the stretch.
                (New distance = stretch * old distance)
            geometry: The initial geometry to perturb.
            atom_pair: A tuple with two integers, indexing which
                atoms from the starting geometry should be stretched apart.
                **Atom1 is stretched away from Atom2, while Atom2 remains
                stationary.**

        Returns:
            end geometry
        """

        def func(curr_dist, extra):
            return curr_dist * extra

        return cls._distance_modifier(func, perturbation, geometry,
                                      atom_pair)

    @classmethod
    def _bend_modifier(cls,
                       function: Callable[[float, float], float],
                       parameter: float,
                       geometry: List[Tuple[str, List[float]]],
                       atom_trio: Tuple[int, int, int]) -> List[Tuple[str, List[float]]]:
        """
        Args:
            function: a function of two parameters (current angle,
                extra parameter) returning the new angle
            parameter: The extra parameter of the function above.
            geometry: The initial geometry to perturb.
            atom_trio: A tuple with three integers, indexing
                which atoms from the starting geometry should be bent apart.
                **Atom1 is bent *away* from Atom3 by an angle whose vertex
                is Atom2, while Atom2 and Atom3 remain stationary.**

        Returns:
            end geometry
        """
        a_1, a_2, a_3 = atom_trio
        starting_coord1 = np.array(geometry[a_1][1])
        coord2 = np.array(geometry[a_2][1])
        coord3 = np.array(geometry[a_3][1])

        distance_vec1to2 = starting_coord1 - coord2
        distance_vec3to2 = coord3 - coord2
        rot_axis = np.cross(distance_vec1to2, distance_vec3to2)
        # If atoms are linear, choose the rotation direction randomly,
        # but still along the correct plane
        # Maybe this is a bad idea if we end up here on some
        # existing bending path.
        # It'd be good to fix this later to remember the axis in some way.
        if np.linalg.norm(rot_axis) == 0:
            nudged_vec = copy.deepcopy(distance_vec1to2)
            nudged_vec[0] += .01
            rot_axis = np.cross(nudged_vec, distance_vec3to2)
        rot_unit_axis = rot_axis / np.linalg.norm(rot_axis)
        starting_angle = np.arcsin(
            np.linalg.norm(rot_axis) / (
                np.linalg.norm(distance_vec1to2)
                * np.linalg.norm(distance_vec3to2)
            )
        )
        new_angle = function(starting_angle, parameter)
        perturbation = new_angle - starting_angle
        rot_matrix = scipy.linalg.expm(
            np.cross(
                np.eye(3),
                rot_unit_axis *
                perturbation))
        new_coord1 = rot_matrix @ starting_coord1

        ending_geometry = copy.deepcopy(geometry)
        ending_geometry[a_1] = ending_geometry[a_1][0], new_coord1.tolist()
        return ending_geometry

    @classmethod
    def absolute_angle(cls,
                       angle: float,
                       geometry: List[Tuple[str, List[float]]],
                       atom_trio: Tuple[int, int, int]) -> List[Tuple[str, List[float]]]:
        """
        Args:
            angle: The magnitude of the perturbation in **radians**.
                **Positive bend is always in the direction toward Atom3.**
                the direction of increasing the starting angle.**
            geometry: The initial geometry to perturb.
            atom_trio: A tuple with three integers, indexing
                which atoms from the starting geometry should be bent apart.
                **Atom1 is bent *away* from Atom3 by an angle whose vertex
                is Atom2 and equal to **angle**, while Atom2 and Atom3
                remain stationary.**

        Returns:
            end geometry
        """

        def func(curr_angle, extra):  # pylint: disable=unused-argument
            return extra

        return cls._bend_modifier(func, angle, geometry, atom_trio)

    @classmethod
    def absolute_bending(cls,
                         bend: float,
                         geometry: List[Tuple[str, List[float]]],
                         atom_trio: Tuple[int, int, int]) -> List[Tuple[str, List[float]]]:
        """
        Args:
            bend: The magnitude of the perturbation in **radians**.
                **Positive bend is always in the direction toward Atom3.**
                the direction of increasing the starting angle.**
            geometry: The initial geometry to perturb.
            atom_trio: A tuple with three integers, indexing
                which atoms from the starting geometry should be bent apart.
                **Atom1 is bent *away* from Atom3 by an angle whose vertex
                is Atom2 and equal to the initial angle **plus** bend,
                while Atom2 and Atom3 remain stationary.**

        Returns:
            end geometry
        """

        def func(curr_angle, extra):
            return curr_angle + extra

        return cls._bend_modifier(func, bend, geometry, atom_trio)

    @classmethod
    def relative_bending(cls,
                         bend: float,
                         geometry: List[Tuple[str, List[float]]],
                         atom_trio: Tuple[int, int, int]) -> List[Tuple[str, List[float]]]:
        """
        Args:
            bend: The magnitude of the perturbation in **radians**.
                **Positive bend is always in the direction toward Atom3.**
                the direction of increasing the starting angle.**
            geometry: The initial geometry to perturb.
            atom_trio: A tuple with three integers,
                indexing which atoms from the starting geometry
                should be bent apart. **Atom1 is bent *away* from Atom3
                by an angle whose vertex is Atom2 and equal to the initial
                angle **times** bend, while Atom2 and Atom3
                remain stationary.**

        Returns:
            end geometry
        """

        def func(curr_angle, extra):
            return curr_angle * extra

        return cls._bend_modifier(func, bend, geometry, atom_trio)

    def _get_perturbed_geom(self) -> List[Tuple[str, List[float]]]:
        """ get perturbed geometry """
        if self.perturbations is None or self._degrees_of_freedom is None:
            return self._geometry

        geometry = copy.deepcopy(self._geometry)
        for per, dof in zip(self.perturbations, self._degrees_of_freedom):
            geometry = dof(per, geometry)
        return geometry

    @property
    def units(self):
        """ The geometry coordinate units """
        return UnitsType.ANGSTROM

    @property
    def geometry(self) -> List[Tuple[str, List[float]]]:
        """ Get geometry accounting for any perturbations """
        return self._get_perturbed_geom()

    @property
    def masses(self) -> Optional[List[float]]:
        """ Get masses """
        return self._masses

    @masses.setter
    def masses(self, value: Optional[List[float]]) -> None:
        """ Set masses
        Args:
            value: masses

        Raises:
            ValueError: Length of masses must match length of geometries.
        """
        Molecule._check_consistency(self._geometry, value)
        self._masses = value

    @property
    def multiplicity(self) -> int:
        """ Get multiplicity """
        return self._multiplicity

    @multiplicity.setter
    def multiplicity(self, value: int) -> None:
        """ Set multiplicity """
        self._multiplicity = value

    @property
    def charge(self) -> int:
        """ Get charge """
        return self._charge

    @charge.setter
    def charge(self, value: int) -> None:
        """ Set charge """
        self._charge = value

    @property
    def perturbations(self) -> Optional[List[float]]:
        """ Get perturbations """
        return self._perturbations

    @perturbations.setter
    def perturbations(self, value: Optional[List[float]]) -> None:
        """ Set perturbations """
        self._perturbations = value
