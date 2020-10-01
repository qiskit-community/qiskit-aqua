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
This module implements an interface for a generic molecule.
It defines the composing atoms (with properties like masses, and nuclear spin),
and allows for changing the molecular geometry through given degrees of freedom
(e.g. bond-stretching, angle-bending, etc.).
"""

from typing import Callable, Tuple, List, Optional
import copy

import numpy as np
import scipy.linalg


class Molecule:
    """
    Molecule class
    """

    def __init__(self,
                 geometry: List[Tuple[str, List[float]]],
                 multiplicity: int,
                 charge: int,
                 degrees_of_freedom: Optional[List[Callable]] = None,
                 masses: Optional[List[float]] = None
                 ) -> None:
        """
        Args:
            geometry: 2d list containing atom string names
                to generate PySCF molecule strings as keys and list of 3
                floats representing Cartesian coordinates as values,
                in units of **Angstroms**.
            multiplicity: multiplicity
            charge: charge
            degrees_of_freedom: List of functions taking a
                perturbation value and geometry and returns a perturbed
                geometry. Helper functions for typical perturbations are
                provided and can be used by the form
                itertools.partial(Molecule.stretching_potential,{'atom_pair': (1, 2))
                to specify the desired degree of freedom.
            masses: masses

        Raises:
            ValueError: Invalid input.
        """
        self._geometry = geometry
        self._degrees_of_freedom = degrees_of_freedom
        self._multiplicity = multiplicity
        self._charge = charge

        if masses is not None and not len(masses) == len(self._geometry):
            raise ValueError(
                'Length of masses must match length of geometries, '
                'found {} and {} respectively'.format(
                    len(masses),
                    len(self._geometry)
                )
            )

        self._masses = masses
        self._basis_set = None  # type: Optional[str]
        self._hf_method = None  # type: Optional[str]

    @classmethod
    def __distance_modifier(cls,
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
        ending_geometry[a_1][1] = new_coord1.tolist()  # type: ignore
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

        return cls.__distance_modifier(func, distance, geometry, atom_pair)

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

        return cls.__distance_modifier(func, perturbation, geometry,
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

        return cls.__distance_modifier(func, perturbation, geometry,
                                       atom_pair)

    @classmethod
    def __bend_modifier(cls,
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
        ending_geometry[a_1][1] = new_coord1.tolist()  # type: ignore
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

        return cls.__bend_modifier(func, angle, geometry, atom_trio)

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

        return cls.__bend_modifier(func, bend, geometry, atom_trio)

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

        return cls.__bend_modifier(func, bend, geometry, atom_trio)

    def get_perturbed_geom(self,
                           perturbations: Optional[List[float]] = None) \
            -> List[Tuple[str, List[float]]]:
        """ get perturbed geometry """
        if not perturbations or not self._degrees_of_freedom:
            return self._geometry
        geometry = copy.deepcopy(self._geometry)
        for per, dof in zip(perturbations, self._degrees_of_freedom):
            geometry = dof(per, geometry)
        return geometry

    @property
    def geometry(self) -> List[Tuple[str, List[float]]]:
        """ return geometry """
        return self._geometry

    @classmethod
    def get_geometry_str(cls,
                         geometry: List[Tuple[str, List[float]]]) -> str:
        """ get geometry string """
        return '; '.join([name + ' ' + ', '.join(map(str, coord))
                          for (name, coord) in geometry])

    @property
    def geometry_str(self) -> str:
        """ return geometry string """
        return Molecule.get_geometry_str(self.geometry)

    @property
    def basis_set(self) -> Optional[str]:
        """ return basis set """
        return self._basis_set

    @basis_set.setter
    def basis_set(self, value: str) -> None:
        """ set basis set """
        self._basis_set = value

    @property
    def hf_method(self) -> Optional[str]:
        """ return hf method """
        return self._hf_method

    @hf_method.setter
    def hf_method(self, value: str) -> None:
        """ set hf method """
        self._hf_method = value

    @property
    def masses(self) -> Optional[List[float]]:
        """ return masses """
        return self._masses

    @property
    def multiplicity(self) -> int:
        """ return multiplicity """
        return self._multiplicity

    @property
    def charge(self) -> int:
        """ return charge """
        return self._charge
