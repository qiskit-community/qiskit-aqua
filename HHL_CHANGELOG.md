# Changelog for HHL module

All the changes done to the qiskit aqua module are documented here.

## Utils

* Added `random_hermitian`, `random_non_hermitian` methods to
    `utils/generate_random_matrix.py`. Containing additional hidden methods
    `random_diag`, `limit_paulis`, `limit_entries`. `random_hermitian,
    random_non_hermitian` are methods for matrix generation with specified
    condition number or eigen/singular values.
* Added `cnx_na.py` for preforming a multiple controlled X gate without using
    ancillary qubits.
* Added `cnu1.py`, `cnu3.py` for multiple controlled U1 and U3 operations.
* Integrated itmes in `__init__.py`.

## Algorithms
### Components

* Added `qfts` as a direct copy of `iqfts` and changed the specific things to
    make it the QFT.
* Added `eigs` component for submodules that find the eigenvales to a matrix
    with input vector.
  - Added `qpe.py` as a one implementation that gets the eigenvalues.
  - Basically copy of `algorithms/single_sample/qpe/qpe.py` with added support
      for negative eigenvalues, inverse circuit construction and evolution time
      instead of operator scaling.
* Added `reciprocals` component for submodules that rotate the reciprocal value
    of a quantum register into the amplitude of an ancillary qubit.
  - Three implementations addded `lookup_rotation.py`, `generated_circuits.py`
      and `longdivision.py`. For details see documentation.
* Make components aqua complient, i.e. adding them in `_quantumalgorithm.py`
    and `__init__` files, respectively.

### Single Sample

* Added `hhl` quantum algorithm for solving linear equations. For details see
    documentation.
