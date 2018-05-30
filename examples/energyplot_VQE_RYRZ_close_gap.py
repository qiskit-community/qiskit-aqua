#import paths
import os
import sys

import matplotlib
matplotlib.use('Agg')
import pylab
from qiskit_acqua_chemistry import ACQUAChemistry
import  argparse
import pprint

import logging
logger = logging.getLogger()
# logger.setLevel(logging.DEBUG)

# README:
# If you want to simply close the gap, just use this command (You still need to tell the distance you are interested in): 
#python3 energyplot_VQE_RYRZ_close_gap.py --distance 1.1
# If you want to specify the arguments fully, use this script:
#python3 energyplot_VQE_RYRZ_close_gap.py --molecule LiH --orbital_reduction 1 --eval_number 20000 --distance 1.1
# Note: this script only allows you to run one point at a time because running multiple points in a script will take impractically long time. You should think carefully to run the scripts in parallel.


# this dictionary is only a template, which is subject to changes in the code below
#note for pyscf:
#  use atomic, rather than atoms.
#  use "spin": 0, rather than multiplicity
acqua_chemistry_dict = {
    "algorithm": {
        "name": "VQE",
        "operator_mode": "matrix"
    },
    "problem":{
         "random_seed": 101
    },
    'backend':{
        'name': 'local_statevector_simulator',
        'shots': 1
    },

    "backend": {
        "name": "local_statevector_simulator"
    },
    "driver": {
        "name": "PYSCF"
    },
    "operator": {
        "name" : "hamiltonian",
        "qubit_mapping": "jordan_wigner"
    },
    "name": [
        "LiH molecule experiment"
    ],
    "optimizer": {
        "maxiter": 3000,
        "name": "COBYLA"
    },
    "pyscf": {
        "atom": "H .0 .0 .0; H .0 .0 0.2",
        "basis": "sto3g",
        "charge": 0,
        "spin": 0,
        "unit": "Angstrom"
    },
    "variational_form": {
        "depth": 10,
        'entanglement': 'linear',
        "name": "RYRZ"
    }


}



# Input dictionary to configure acqua_chemistry for the chemistry problem.
# Note: In order to allow this to run reasonably quickly it takes advantage
#       of the ability to freeze core orbitals and remove unoccupied virtual
#       orbitals to reduce the size of the problem. The result without this
#       will be more accurate but it takes rather longer to run.

#  LiH: python3 energyplot_VQE_RYRZ_close_gap.py --molecule LiH --eval_number 10000 --distance 1.0 --optimizer COBYLA --noisy True
def makeArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument('--optimizer', type=str, default="COBYLA",
                           help='optimizer name')
    parser.add_argument('--distance', type=float, default=1.0,
                           help='steps')
    parser.add_argument('--molecule', type=str, default="LiH",
                           help='molecular')
    parser.add_argument('--eval_number', type=int, default=1000,
                           help='number of eval')
    parser.add_argument('--initial_point_seed', type=int, default=100,
                           help='seed for random init point generation')
    parser.add_argument('--noisy', type=bool, default=False,
                           help='do we have noise?')

    args = parser.parse_args()
    return args





def report(distances, energies, args):
    print('Distances: ', distances)
    print('Energies:', energies)
    pylab.plot(distances, energies)
    pylab.xlabel('Interatomic distance')
    pylab.ylabel('Energy')
    pylab.title('LiH Ground State Energy')
    pylab.savefig(args.molecule + "_VQE")



if __name__ == '__main__':




    args = makeArgs()
    depths = {
        'H2': 10,
        'LiH': 10
    }
    evalnums = {
        'H2': 10000,
        'LiH': 10000
    }

    molecule_templates = {
        'H2': 'H .0 .0 -{0}; H .0 .0 {0}',
        'LiH': 'Li .0 .0 -{0}; H .0 .0 {0}'
    }

    starts = {
        'H2': 0.2,
        'LiH': 1.25
    }
    bys= {
        'H2': 1.2,
        'LiH': 0.5
    }

    acqua_chemistry_dict['problem']['random_seed'] = args.initial_point_seed
    acqua_chemistry_dict['optimizer']['name'] = args.optimizer
    acqua_chemistry_dict['variational_form']['depth'] = depths[args.molecule]
    acqua_chemistry_dict['optimizer']['maxiter'] = evalnums[args.molecule]


    if args.eval_number > 0:
        acqua_chemistry_dict['optimizer']['maxiter'] = args.eval_number


    if args.molecule == 'LiH':
        acqua_chemistry_dict['operator']['qubit_mapping'] = 'parity'
        acqua_chemistry_dict['operator']['two_qubit_reduction'] = True

        acqua_chemistry_dict['operator']['freeze_core'] = True
    elif args.molecule == 'H2': # no, we cannot reduce for H2
        pass


    if args.noisy:
        acqua_chemistry_dict['backend']['name'] = 'local_qasm_simulator'
        acqua_chemistry_dict['backend']['shots'] = 1000



    molecule = molecule_templates[args.molecule]
    acqua_chemistry_dict['pyscf']['atom'] = molecule  # temporarily set, will be overwritten

    start = starts[args.molecule]
    by    = bys[args.molecule] # How much to increase distance by



    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(acqua_chemistry_dict)


    #print('\b\b{:2d}'.format(i), end='', flush=True)
    d = args.distance
    acqua_chemistry_dict['pyscf']['atom'] = molecule.format(d / 2)
    solver = ACQUAChemistry()
    result = solver.run(acqua_chemistry_dict)
    print(d, result['energy'], result['total_dipole_moment'])



    # the output will be appended to a file
    with open('./' + args.molecule +  '_distance='+ str(args.distance) + "_optimizer=" + str(args.optimizer), 'w') as f:
        f.write("\ndistance: " + str(d) +"\n")
        f.write("energy:" + str(result['energy'])+"\n")
        f.write("dipole moment:" + str(result['total_dipole_moment'])+"\n")
        f.write("\n")
        for line in result['printable']:
            f.write(line + "\n")
