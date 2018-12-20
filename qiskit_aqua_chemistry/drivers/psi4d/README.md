# Qiskit Chemistry

## Electronic structure driver for PSI4

PSI4 is an open-source program for computational chemistry, see http://www.psicode.org/ for downloads and its
licensing terms.

This driver requires PSI4 to be installed and available for Qiskit Chemistry to access/run. Once download and
installed the executable psi4 should be on the Path. If not make sure that it is so the driver can find the
psi4 executable

## Input file example
To configure a molecule on which to do a chemistry experiment with Qiskit Chemistry create a PSI4 section in the
input file as per the example below. Here the molecule, basis set and other options are specified according to PSI4 
```
&PSI4
molecule h2 {
   0 1
   H 0.0 0.0 0.0
   H 0.0 0.0 0.74
}

set {
  basis sto-3g
  scf_type pk
}
&END
```
