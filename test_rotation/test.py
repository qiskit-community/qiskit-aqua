from qiskit_aqua.algorithms.single_sample.hhl.lookup_rotation import LUP_ROTATION

LUP_ROTATION.test_value_range(6,5)

print(LUP_ROTATION.classic_approx(5,4,2,1))#False))
#print(LUP_ROTATION.get_initial_statevector_representation(['1','0','1']))
#print(LUP_ROTATION.get_complete_statevector_representation(['1','0','1'],0.25))
