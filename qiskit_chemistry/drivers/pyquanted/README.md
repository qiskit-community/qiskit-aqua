# Qiskit Chemistry

## Electronic structure driver for PyQuante2

PyQuante2 is an open-source library for computational chemistry, see https://github.com/rpmuller/pyquante2 for
installation instructions and its licensing terms. 

This driver contains a couple of methods here, in transform.py, from Pyquante1, which was licensed under a
[modified BSD license](./LICENSE.txt) 

This driver requires PyQuante2 to be installed and available for Qiskit Chemistry to access/call.

_**Note**: molecular dipole moment is not computed by Qiskit Chemistry when using this driver._

## Input file example
To configure a molecule on which to do a chemistry experiment with Qiskit Chemistry create a PYQUANTE section
in the input file as per the example below. Here the molecule, basis set and other options are specified as
key value pairs. The molecule is a list of atoms in xyz coords separated by semi-colons ';'.  
```
&PYQUANTE
  atoms=H .0 .0 .0; H .0 .0 0.74
  units=Angstrom
  charge=0
  multiplicity=1
  basis=sto3g
&END
```
