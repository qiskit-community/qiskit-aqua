## Particle-hole Hamiltonian 

The 'standard' second quantized Hamiltonian can be transformed in the particle-hole (p/h) picture, which makes the
expansion of the trail wavefunction from the HF reference state more natural. In fact, for both trail wavefunctions
implemented in q-lib ('heuristic' hardware efficient and UCCSD) the p/h Hamiltonian improves the speed of convergence of the
VQE algorithm for the calculation of the electronic ground state properties.
For more information on the p/h formalism see: [P. Barkoutsos, arXiv:1805.04340](https://arxiv.org/abs/1805.04340).

Programmatically, to enable calculations with the p/h Hamiltonian set:

`ferOp = FermionicOperator(h1=molecule._one_body_integrals, h2=molecule._two_body_integrals)
newferOp, energy_shift = ferOp.particle_hole_transformation(num_particles=2)`
