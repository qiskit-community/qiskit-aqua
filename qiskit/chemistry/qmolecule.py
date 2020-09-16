# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" QMolecule """

from typing import List
import os
import logging
import tempfile
import warnings
import numpy
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import h5py

logger = logging.getLogger(__name__)


class QMolecule:
    """
    Molecule data class containing driver result.

    When one of the chemistry :mod:`~qiskit.chemistry.drivers` is run and instance
    of this class is returned. This contains various properties that are made available in
    a consistent manner across the various drivers.

    Note that values here, for the same input molecule to each driver, may be vary across
    the drivers underlying code implementation. Also some drivers may not provide certain fields
    such as dipole integrals in the case of :class:`~qiskit.chemistry.drivers.PyQuanteDriver`.

    This class provides methods to save it and load it again from an HDF5 file
    """

    QMOLECULE_VERSION = 2

    def __init__(self, filename=None):
        self._filename = filename

        # All the following fields are saved/loaded in the save/load methods.
        # If fields are added in a version they are noted by version comment
        #
        # Originally support was limited to closed shell, when open shell was
        # added, and new integrals to allow different Beta orbitals needed,
        # these have been added as new similarly named fields but with suffices
        # such as _b, _bb and _ba. So mo_coeff (with no suffix) is the original
        # and is for alpha molecular coefficients, the added one for beta is
        # name mo_coeff_b, i.e. same name but with _b suffix. To keep backward
        # compatibility the original fields were not renamed with an _a suffix
        # but rather its implicit in the lack thereof given another field of
        # the same name but with an explicit suffix.

        # Driver origin from which this QMolecule was created
        self.origin_driver_name = "?"
        self.origin_driver_version = "?"  # v2
        self.origin_driver_config = "?"

        # Energies and orbits
        self.hf_energy = None
        self.nuclear_repulsion_energy = None
        self.num_orbitals = None
        self.num_alpha = None
        self.num_beta = None
        self.mo_coeff = None
        self.mo_coeff_b = None  # v2
        self.orbital_energies = None
        self.orbital_energies_b = None  # v2

        # Molecule geometry. xyz coords are in Bohr
        self.molecular_charge = None
        self.multiplicity = None
        self.num_atoms = None
        self.atom_symbol = None
        self.atom_xyz = None

        # 1 and 2 electron ints in AO basis
        self.hcore = None  # v2
        self.hcore_b = None  # v2
        self.kinetic = None  # v2
        self.overlap = None  # v2
        self.eri = None  # v2

        # 1 and 2 electron integrals in MO basis
        self.mo_onee_ints = None
        self.mo_onee_ints_b = None  # v2
        self.mo_eri_ints = None
        self.mo_eri_ints_bb = None  # v2
        self.mo_eri_ints_ba = None  # v2

        # Dipole moment integrals in AO basis
        self.x_dip_ints = None  # v2
        self.y_dip_ints = None  # v2
        self.z_dip_ints = None  # v2

        # Dipole moment integrals in MO basis
        self.x_dip_mo_ints = None
        self.x_dip_mo_ints_b = None  # v2
        self.y_dip_mo_ints = None
        self.y_dip_mo_ints_b = None  # v2
        self.z_dip_mo_ints = None
        self.z_dip_mo_ints_b = None  # v2
        self.nuclear_dipole_moment = None
        self.reverse_dipole_sign = False

    @property
    def one_body_integrals(self):
        """ Returns one body electron integrals. """
        return QMolecule.onee_to_spin(self.mo_onee_ints, self.mo_onee_ints_b)

    @property
    def two_body_integrals(self):
        """ Returns two body electron integrals. """
        return QMolecule.twoe_to_spin(self.mo_eri_ints, self.mo_eri_ints_bb, self.mo_eri_ints_ba)

    def has_dipole_integrals(self):
        """ Check if dipole integrals are present. """
        return self.x_dip_mo_ints is not None and \
            self.y_dip_mo_ints is not None and \
            self.z_dip_mo_ints is not None

    @property
    def x_dipole_integrals(self):
        """ returns x_dipole_integrals """
        return QMolecule.onee_to_spin(self.x_dip_mo_ints, self.x_dip_mo_ints_b)

    @property
    def y_dipole_integrals(self):
        """ returns y_dipole_integrals """
        return QMolecule.onee_to_spin(self.y_dip_mo_ints, self.y_dip_mo_ints_b)

    @property
    def z_dipole_integrals(self):
        """ returns z_dipole_integrals """
        return QMolecule.onee_to_spin(self.z_dip_mo_ints, self.z_dip_mo_ints_b)

    def Z(self, natom):  # pylint: disable=invalid-name
        """ Z """
        if natom < 0 or natom >= self.num_atoms:
            raise ValueError("Atom index out of range")
        return QMolecule.symbols.index(self.atom_symbol[natom].lower().capitalize())

    @property
    def core_orbitals(self) -> List[int]:
        """
        Returns:
            A list of core orbital indices.
        """
        if self.num_atoms is None:
            logger.warning("Missing molecule information! Returning empty core orbital list.")
            return []
        count = 0
        for i in range(self.num_atoms):
            z = self.Z(i)
            if z > 2:
                count += 1
            if z > 10:
                count += 4
            if z > 18:
                count += 4
            if z > 36:
                count += 9
            if z > 54:
                count += 9
            if z > 86:
                count += 16
        return list(range(count))

    @property
    def filename(self):
        """ returns temp file path """
        if self._filename is None:
            file, self._filename = tempfile.mkstemp(suffix='.hdf5')
            os.close(file)

        return self._filename

    def load(self):
        """loads info saved."""
        try:
            if self._filename is None:
                return

            with h5py.File(self._filename, "r") as file:
                def read_array(name):
                    _data = file[name][...]
                    if _data.dtype == numpy.bool and _data.size == 1 and not _data:
                        _data = None
                    return _data

                # A version field was added to save format from version 2 so if
                # there is no version then we have original (version 1) format
                version = 1
                if 'version' in file.keys():
                    data = file["version"][...]
                    version = int(data) if data.dtype.num != 0 else version

                # Origin driver info
                data = file["origin_driver/name"][...]
                self.origin_driver_name = data[...].tobytes().decode('utf-8')
                self.origin_driver_version = '?'
                if version > 1:
                    data = file["origin_driver/version"][...]
                    self.origin_driver_version = data[...].tobytes().decode('utf-8')
                data = file["origin_driver/config"][...]
                self.origin_driver_config = data[...].tobytes().decode('utf-8')

                # Energies
                data = file["energy/hf_energy"][...]
                self.hf_energy = float(data) if data.dtype.num != 0 else None
                data = file["energy/nuclear_repulsion_energy"][...]
                self.nuclear_repulsion_energy = float(data) if data.dtype.num != 0 else None

                # Orbitals
                data = file["orbitals/num_orbitals"][...]
                self.num_orbitals = int(data) if data.dtype.num != 0 else None
                data = file["orbitals/num_alpha"][...]
                self.num_alpha = int(data) if data.dtype.num != 0 else None
                data = file["orbitals/num_beta"][...]
                self.num_beta = int(data) if data.dtype.num != 0 else None
                self.mo_coeff = read_array("orbitals/mo_coeff")
                self.mo_coeff_b = read_array("orbitals/mo_coeff_B") if version > 1 else None
                self.orbital_energies = read_array("orbitals/orbital_energies")
                self.orbital_energies_b = \
                    read_array("orbitals/orbital_energies_B") if version > 1 else None

                # Molecule geometry
                data = file["geometry/molecular_charge"][...]
                self.molecular_charge = int(data) if data.dtype.num != 0 else None
                data = file["geometry/multiplicity"][...]
                self.multiplicity = int(data) if data.dtype.num != 0 else None
                data = file["geometry/num_atoms"][...]
                self.num_atoms = int(data) if data.dtype.num != 0 else None
                data = file["geometry/atom_symbol"][...]
                self.atom_symbol = [a.decode('utf8') for a in data]
                self.atom_xyz = file["geometry/atom_xyz"][...]

                # 1 and 2 electron integrals in AO basis
                self.hcore = read_array("integrals/hcore") if version > 1 else None
                self.hcore_b = read_array("integrals/hcore_B") if version > 1 else None
                self.kinetic = read_array("integrals/kinetic") if version > 1 else None
                self.overlap = read_array("integrals/overlap") if version > 1 else None
                self.eri = read_array("integrals/eri") if version > 1 else None

                # 1 and 2 electron integrals in MO basis
                self.mo_onee_ints = read_array("integrals/mo_onee_ints")
                self.mo_onee_ints_b = \
                    read_array("integrals/mo_onee_ints_B") if version > 1 else None
                self.mo_eri_ints = read_array("integrals/mo_eri_ints")
                self.mo_eri_ints_bb = \
                    read_array("integrals/mo_eri_ints_BB") if version > 1 else None
                self.mo_eri_ints_ba = \
                    read_array("integrals/mo_eri_ints_BA") if version > 1 else None

                # dipole integrals in AO basis
                self.x_dip_ints = read_array("dipole/x_dip_ints") if version > 1 else None
                self.y_dip_ints = read_array("dipole/y_dip_ints") if version > 1 else None
                self.z_dip_ints = read_array("dipole/z_dip_ints") if version > 1 else None

                # dipole integrals in MO basis
                self.x_dip_mo_ints = read_array("dipole/x_dip_mo_ints")
                self.x_dip_mo_ints_b = read_array("dipole/x_dip_mo_ints_B") if version > 1 else None
                self.y_dip_mo_ints = read_array("dipole/y_dip_mo_ints")
                self.y_dip_mo_ints_b = read_array("dipole/y_dip_mo_ints_B") if version > 1 else None
                self.z_dip_mo_ints = read_array("dipole/z_dip_mo_ints")
                self.z_dip_mo_ints_b = read_array("dipole/z_dip_mo_ints_B") if version > 1 else None
                self.nuclear_dipole_moment = file["dipole/nuclear_dipole_moment"][...]
                self.reverse_dipole_sign = file["dipole/reverse_dipole_sign"][...]

        except OSError:
            pass

    def save(self, file_name=None):
        """Saves the info from the driver."""
        file = None
        if file_name is not None:
            self.remove_file(file_name)
            file = file_name
        else:
            file = self.filename
            self.remove_file()

        with h5py.File(file, "w") as file:
            def create_dataset(group, name, value):
                def is_float(v):
                    if v is None:
                        return False
                    if isinstance(v, float):
                        return True
                    if isinstance(v, list):
                        v = numpy.array(v)

                    try:
                        return numpy.issubdtype(v.dtype, numpy.floating)
                    except Exception:  # pylint: disable=broad-except
                        return False

                if is_float(value):
                    group.create_dataset(name, data=value, dtype="float64")
                else:
                    group.create_dataset(name, data=(value if value is not None else False))

            file.create_dataset("version", data=(self.QMOLECULE_VERSION,))

            # Driver origin of molecule data
            g_driver = file.create_group("origin_driver")
            g_driver.create_dataset(
                "name", data=(numpy.string_(self.origin_driver_name)
                              if self.origin_driver_name is not None else numpy.string_("?")))
            g_driver.create_dataset(
                "version", data=(numpy.string_(self.origin_driver_version)
                                 if self.origin_driver_version is not None else numpy.string_("?")))
            g_driver.create_dataset(
                "config", data=(numpy.string_(self.origin_driver_config)
                                if self.origin_driver_config is not None else numpy.string_("?")))

            # Energies
            g_energy = file.create_group("energy")
            create_dataset(g_energy, "hf_energy", self.hf_energy)
            create_dataset(g_energy, "nuclear_repulsion_energy", self.nuclear_repulsion_energy)

            # Orbitals
            g_orbitals = file.create_group("orbitals")
            create_dataset(g_orbitals, "num_orbitals", self.num_orbitals)
            create_dataset(g_orbitals, "num_alpha", self.num_alpha)
            create_dataset(g_orbitals, "num_beta", self.num_beta)
            create_dataset(g_orbitals, "mo_coeff", self.mo_coeff)
            create_dataset(g_orbitals, "mo_coeff_B", self.mo_coeff_b)
            create_dataset(g_orbitals, "orbital_energies", self.orbital_energies)
            create_dataset(g_orbitals, "orbital_energies_B", self.orbital_energies_b)

            # Molecule geometry
            g_geometry = file.create_group("geometry")
            create_dataset(g_geometry, "molecular_charge", self.molecular_charge)
            create_dataset(g_geometry, "multiplicity", self.multiplicity)
            create_dataset(g_geometry, "num_atoms", self.num_atoms)
            g_geometry.create_dataset(
                "atom_symbol", data=([a.encode('utf8') for a in self.atom_symbol]
                                     if self.atom_symbol is not None else False))
            create_dataset(g_geometry, "atom_xyz", self.atom_xyz)

            # 1 and 2 electron integrals
            g_integrals = file.create_group("integrals")
            create_dataset(g_integrals, "hcore", self.hcore)
            create_dataset(g_integrals, "hcore_B", self.hcore_b)
            create_dataset(g_integrals, "kinetic", self.kinetic)
            create_dataset(g_integrals, "overlap", self.overlap)
            create_dataset(g_integrals, "eri", self.eri)
            create_dataset(g_integrals, "mo_onee_ints", self.mo_onee_ints)
            create_dataset(g_integrals, "mo_onee_ints_B", self.mo_onee_ints_b)
            create_dataset(g_integrals, "mo_eri_ints", self.mo_eri_ints)
            create_dataset(g_integrals, "mo_eri_ints_BB", self.mo_eri_ints_bb)
            create_dataset(g_integrals, "mo_eri_ints_BA", self.mo_eri_ints_ba)

            # dipole integrals
            g_dipole = file.create_group("dipole")
            create_dataset(g_dipole, "x_dip_ints", self.x_dip_ints)
            create_dataset(g_dipole, "y_dip_ints", self.y_dip_ints)
            create_dataset(g_dipole, "z_dip_ints", self.z_dip_ints)
            create_dataset(g_dipole, "x_dip_mo_ints", self.x_dip_mo_ints)
            create_dataset(g_dipole, "x_dip_mo_ints_B", self.x_dip_mo_ints_b)
            create_dataset(g_dipole, "y_dip_mo_ints", self.y_dip_mo_ints)
            create_dataset(g_dipole, "y_dip_mo_ints_B", self.y_dip_mo_ints_b)
            create_dataset(g_dipole, "z_dip_mo_ints", self.z_dip_mo_ints)
            create_dataset(g_dipole, "z_dip_mo_ints_B", self.z_dip_mo_ints_b)
            create_dataset(g_dipole, "nuclear_dipole_moment", self.nuclear_dipole_moment)
            create_dataset(g_dipole, "reverse_dipole_sign", self.reverse_dipole_sign)

    def remove_file(self, file_name=None):
        """ remove file """
        try:
            file = self._filename if file_name is None else file_name
            os.remove(file)
        except OSError:
            pass

    # Utility functions to convert integrals into the form expected by Qiskit's chemistry module

    @staticmethod
    def oneeints2mo(ints, moc):
        """Converts one-body integrals from AO to MO basis

        Returns one electron integrals in AO basis converted to given MO basis

        Args:
            ints (numpy.ndarray): N^2 one electron integrals in AO basis
            moc (numpy.ndarray): Molecular orbital coefficients
        Returns:
            numpy.ndarray: integrals in MO basis
        """
        return numpy.dot(numpy.dot(numpy.transpose(moc), ints), moc)

    @staticmethod
    def twoeints2mo(ints, moc):
        """Converts two-body integrals from AO to MO basis

        Returns two electron integrals in AO basis converted to given MO basis

        Args:
            ints (numpy.ndarray): N^2 two electron integrals in AO basis
            moc (numpy.ndarray): Molecular orbital coefficients

        Returns:
            numpy.ndarray: integrals in MO basis
        """
        dim = ints.shape[0]
        eri_mo = numpy.zeros((dim, dim, dim, dim))

        for a_i in range(dim):
            temp1 = numpy.einsum('i,i...->...', moc[:, a_i], ints)
            for b_i in range(dim):
                temp2 = numpy.einsum('j,j...->...', moc[:, b_i], temp1)
                temp3 = numpy.einsum('kc,k...->...c', moc, temp2)
                eri_mo[a_i, b_i, :, :] = numpy.einsum('ld,l...c->...cd', moc, temp3)

        return eri_mo

    @staticmethod
    def twoeints2mo_general(ints, moc1, moc2, moc3, moc4):
        """ twoeints2mo_general """
        return numpy.einsum('pqrs,pi,qj,rk,sl->ijkl', ints, moc1, moc2, moc3, moc4)

    @staticmethod
    def onee_to_spin(mohij, mohij_b=None, threshold=1E-12):
        """Convert one-body MO integrals to spin orbital basis

        Takes one body integrals in molecular orbital basis and returns
        integrals in spin orbitals ready for use as coefficients to
        one body terms 2nd quantized Hamiltonian.

        Args:
            mohij (numpy.ndarray): One body orbitals in molecular basis (Alpha)
            mohij_b (numpy.ndarray): One body orbitals in molecular basis (Beta)
            threshold (float): Threshold value for assignments
        Returns:
            numpy.ndarray: One body integrals in spin orbitals
        """
        if mohij_b is None:
            mohij_b = mohij

        # The number of spin orbitals is twice the number of orbitals
        norbs = mohij.shape[0]
        nspin_orbs = 2*norbs

        # One electron terms
        moh1_qubit = numpy.zeros([nspin_orbs, nspin_orbs])
        for p in range(nspin_orbs):  # pylint: disable=invalid-name
            for q in range(nspin_orbs):
                spinp = int(p/norbs)
                spinq = int(q/norbs)
                if spinp % 2 != spinq % 2:
                    continue
                ints = mohij if spinp == 0 else mohij_b
                orbp = int(p % norbs)
                orbq = int(q % norbs)
                if abs(ints[orbp, orbq]) > threshold:
                    moh1_qubit[p, q] = ints[orbp, orbq]

        return moh1_qubit

    @staticmethod
    def twoe_to_spin(mohijkl, mohijkl_bb=None, mohijkl_ba=None, threshold=1E-12):
        """Convert two-body MO integrals to spin orbital basis

        Takes two body integrals in molecular orbital basis and returns
        integrals in spin orbitals ready for use as coefficients to
        two body terms in 2nd quantized Hamiltonian.

        Args:
            mohijkl (numpy.ndarray): Two body orbitals in molecular basis (AlphaAlpha)
            mohijkl_bb (numpy.ndarray): Two body orbitals in molecular basis (BetaBeta)
            mohijkl_ba (numpy.ndarray): Two body orbitals in molecular basis (BetaAlpha)
            threshold (float): Threshold value for assignments
        Returns:
            numpy.ndarray: Two body integrals in spin orbitals
        """
        ints_aa = numpy.einsum('ijkl->ljik', mohijkl)

        if mohijkl_bb is None or mohijkl_ba is None:
            ints_bb = ints_ba = ints_ab = ints_aa
        else:
            ints_bb = numpy.einsum('ijkl->ljik', mohijkl_bb)
            ints_ba = numpy.einsum('ijkl->ljik', mohijkl_ba)
            ints_ab = numpy.einsum('ijkl->ljik', mohijkl_ba.transpose())

        # The number of spin orbitals is twice the number of orbitals
        norbs = mohijkl.shape[0]
        nspin_orbs = 2*norbs

        # The spin orbitals are mapped in the following way:
        #       Orbital zero, spin up mapped to qubit 0
        #       Orbital one,  spin up mapped to qubit 1
        #       Orbital two,  spin up mapped to qubit 2
        #            .
        #            .
        #       Orbital zero, spin down mapped to qubit norbs
        #       Orbital one,  spin down mapped to qubit norbs+1
        #            .
        #            .
        #            .

        # Two electron terms
        moh2_qubit = numpy.zeros([nspin_orbs, nspin_orbs, nspin_orbs, nspin_orbs])
        for p in range(nspin_orbs):  # pylint: disable=invalid-name
            for q in range(nspin_orbs):
                for r in range(nspin_orbs):
                    for s in range(nspin_orbs):  # pylint: disable=invalid-name
                        spinp = int(p/norbs)
                        spinq = int(q/norbs)
                        spinr = int(r/norbs)
                        spins = int(s/norbs)
                        if spinp != spins:
                            continue
                        if spinq != spinr:
                            continue
                        if spinp == 0:
                            ints = ints_aa if spinq == 0 else ints_ba
                        else:
                            ints = ints_ab if spinq == 0 else ints_bb
                        orbp = int(p % norbs)
                        orbq = int(q % norbs)
                        orbr = int(r % norbs)
                        orbs = int(s % norbs)
                        if abs(ints[orbp, orbq, orbr, orbs]) > threshold:
                            moh2_qubit[p, q, r, s] = -0.5*ints[orbp, orbq, orbr, orbs]

        return moh2_qubit

    symbols = [
        # pylint: disable=bad-option-value,bad-whitespace
        '_',
        'H',  'He',
        'Li', 'Be', 'B',  'C',  'N',  'O',  'F',  'Ne',
        'Na', 'Mg', 'Al', 'Si', 'P',  'S',  'Cl', 'Ar',
        'K',  'Ca', 'Sc', 'Ti', 'V',  'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga',
        'Ge', 'As', 'Se', 'Br', 'Kr',
        'Rb', 'Sr', 'Y',  'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In',
        'Sn', 'Sb', 'Te', 'I', 'Xe',
        'Cs', 'Ba',
        'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
        'Hf', 'Ta', 'W',  'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',
        'Fr', 'Ra',
        'Ac', 'Th', 'Pa', 'U',  'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
        'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']

    BOHR = 0.52917721092  # No of Angstroms in Bohr (from 2010 CODATA)
    DEBYE = 0.393430307   # No ea0 in Debye. Use to convert our dipole moment numbers to Debye

    def log(self):
        """ log properties """
        if not logger.isEnabledFor(logging.INFO):
            return
        opts = numpy.get_printoptions()
        try:
            numpy.set_printoptions(precision=8, suppress=True)

            # Originating driver name & config if set
            if self.origin_driver_name and self.origin_driver_name != "?":
                logger.info("Originating driver name: %s", self.origin_driver_name)
                logger.info("Originating driver version: %s", self.origin_driver_version)
                logger.info("Originating driver config:\n%s", self.origin_driver_config[:-1])

            logger.info("Computed Hartree-Fock energy: %s", self.hf_energy)
            logger.info("Nuclear repulsion energy: %s", self.nuclear_repulsion_energy)
            if None not in (self.hf_energy, self.nuclear_repulsion_energy):
                logger.info("One and two electron Hartree-Fock energy: %s",
                            self.hf_energy - self.nuclear_repulsion_energy)
            logger.info("Number of orbitals is %s", self.num_orbitals)
            logger.info("%s alpha and %s beta electrons", self.num_alpha, self.num_beta)
            logger.info("Molecule comprises %s atoms and in xyz format is ::", self.num_atoms)
            logger.info("  %s, %s", self.molecular_charge, self.multiplicity)
            if self.num_atoms is not None:
                for n in range(0, self.num_atoms):
                    logger.info("  %2s  %s, %s, %s", self.atom_symbol[n],
                                self.atom_xyz[n][0] * QMolecule.BOHR,
                                self.atom_xyz[n][1] * QMolecule.BOHR,
                                self.atom_xyz[n][2] * QMolecule.BOHR)
            if self.mo_coeff is not None:
                logger.info("MO coefficients A: %s", self.mo_coeff.shape)
                logger.debug("\n%s", self.mo_coeff)
            if self.mo_coeff_b is not None:
                logger.info("MO coefficients B: %s", self.mo_coeff_b.shape)
                logger.debug("\n%s", self.mo_coeff_b)
            if self.orbital_energies is not None:
                logger.info("Orbital energies A: %s", self.orbital_energies)
            if self.orbital_energies_b is not None:
                logger.info("Orbital energies B: %s", self.orbital_energies_b)

            if self.hcore is not None:
                logger.info("hcore integrals: %s", self.hcore.shape)
                logger.debug("\n%s", self.hcore)
            if self.hcore_b is not None:
                logger.info("hcore Beta integrals: %s", self.hcore_b.shape)
                logger.debug("\n%s", self.hcore_b)
            if self.kinetic is not None:
                logger.info("kinetic integrals: %s", self.kinetic.shape)
                logger.debug("\n%s", self.kinetic)
            if self.overlap is not None:
                logger.info("overlap integrals: %s", self.overlap.shape)
                logger.debug("\n%s", self.overlap)
            if self.eri is not None:
                logger.info("eri integrals: %s", self.eri.shape)
                logger.debug("\n%s", self.eri)

            if self.mo_onee_ints is not None:
                logger.info("One body MO A integrals: %s", self.mo_onee_ints.shape)
                logger.debug("\n%s", self.mo_onee_ints)
            if self.mo_onee_ints_b is not None:
                logger.info("One body MO B integrals: %s", self.mo_onee_ints_b.shape)
                logger.debug(self.mo_onee_ints_b)

            if self.mo_eri_ints is not None:
                logger.info("Two body ERI MO AA integrals: %s", self.mo_eri_ints.shape)
                logger.debug("\n%s", self.mo_eri_ints)
            if self.mo_eri_ints_bb is not None:
                logger.info("Two body ERI MO BB integrals: %s", self.mo_eri_ints_bb.shape)
                logger.debug("\n%s", self.mo_eri_ints_bb)
            if self.mo_eri_ints_ba is not None:
                logger.info("Two body ERI MO BA integrals: %s", self.mo_eri_ints_ba.shape)
                logger.debug("\n%s", self.mo_eri_ints_ba)

            if self.x_dip_ints is not None:
                logger.info("x dipole integrals: %s", self.x_dip_ints.shape)
                logger.debug("\n%s", self.x_dip_ints)
            if self.y_dip_ints is not None:
                logger.info("y dipole integrals: %s", self.y_dip_ints.shape)
                logger.debug("\n%s", self.y_dip_ints)
            if self.z_dip_ints is not None:
                logger.info("z dipole integrals: %s", self.z_dip_ints.shape)
                logger.debug("\n%s", self.z_dip_ints)

            if self.x_dip_mo_ints is not None:
                logger.info("x dipole MO A integrals: %s", self.x_dip_mo_ints.shape)
                logger.debug("\n%s", self.x_dip_mo_ints)
            if self.x_dip_mo_ints_b is not None:
                logger.info("x dipole MO B integrals: %s", self.x_dip_mo_ints_b.shape)
                logger.debug("\n%s", self.x_dip_mo_ints_b)
            if self.y_dip_mo_ints is not None:
                logger.info("y dipole MO A integrals: %s", self.y_dip_mo_ints.shape)
                logger.debug("\n%s", self.y_dip_mo_ints)
            if self.y_dip_mo_ints_b is not None:
                logger.info("y dipole MO B integrals: %s", self.y_dip_mo_ints_b.shape)
                logger.debug("\n%s", self.y_dip_mo_ints_b)
            if self.z_dip_mo_ints is not None:
                logger.info("z dipole MO A integrals: %s", self.z_dip_mo_ints.shape)
                logger.debug("\n%s", self.z_dip_mo_ints)
            if self.z_dip_mo_ints_b is not None:
                logger.info("z dipole MO B integrals: %s", self.z_dip_mo_ints_b.shape)
                logger.debug("\n%s", self.z_dip_mo_ints_b)

            if self.nuclear_dipole_moment is not None:
                logger.info("Nuclear dipole moment: %s", self.nuclear_dipole_moment)
            if self.reverse_dipole_sign is not None:
                logger.info("Reversal of electronic dipole moment sign needed: %s",
                            self.reverse_dipole_sign)

            logger.info("Core orbitals list %s", self.core_orbitals)
        finally:
            numpy.set_printoptions(**opts)
