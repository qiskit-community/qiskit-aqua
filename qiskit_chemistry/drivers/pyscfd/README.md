# Qiskit Chemistry

## Electronic structure driver for PySCF

PySCF is an open-source library for computational chemistry, see https://github.com/sunqm/pyscf for further
information and its license. The [documentation](http://sunqm.github.io/pyscf/index.html) for PySCF can be
referred to for comprehensive [installation](http://sunqm.github.io/pyscf/install.html) instructions.

This driver requires PySCF to be installed and available for Qiskit Chemistry to access/call.

## Input file example
To configure a molecule on which to do a chemistry experiment with Qiskit Chemistry create a PYSCF section in the
input file as per the example below. Here the molecule, basis set and other options are specified as key value pairs. 
Configuration supported here is a subset of the arguments as can be passed to PySCF pyscf.gto.Mole class namely:
*atom (str only), unit, charge, spin, basis (str only)*.
*max_memory* may be specified here to override PySCF default and should be specified the same way
i.e in MB e.g 4000 for 4GB
```
&PYSCF
  atom=H .0 .0 .0; H .0 .0 0.74
  unit=Angstrom
  charge=0
  spin=0
  basis=sto3g
&END
```
