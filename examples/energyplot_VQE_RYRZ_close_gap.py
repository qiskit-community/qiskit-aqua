#import paths
import os
import sys
algo_directory = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0,algo_directory)
import matplotlib
matplotlib.use('Agg')
import pylab
from qiskit_acqua_chemistry import QISChem
import  argparse
import pprint



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
qischem_dict =  {
    "algorithm": {
        "name": "VQE",
        "operator_mode": "matrix",
        "shots": 1
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
        "entangler_map": {
            "0": [
                1
            ]
        },
        "name": "RYRZ"
    }


}



# Input dictionary to configure qischem for the chemistry problem.
# Note: In order to allow this to run reasonably quickly it takes advantage
#       of the ability to freeze core orbitals and remove unoccupied virtual
#       orbitals to reduce the size of the problem. The result without this
#       will be more accurate but it takes rather longer to run.

#  LiH: --molecule LiH --orbital_reduction 1 --eval_number 10000 --distance 1.0
def makeArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument('--optimizer', type=str, default="COBYLA",
                           help='optimizer name')
    parser.add_argument('--distance', type=float, default=1.0,
                           help='steps')
    parser.add_argument('--molecule', type=str, default="LiH",
                           help='molecular')
    parser.add_argument('--orbital_reduction', type=int, default=1,
                           help='orbital reduction')
    parser.add_argument('--map', type=str, default="linear", # all
                           help='orbital reduction')
    parser.add_argument('--eval_number', type=int, default=20000,
                           help='orbital reduction')

    args = parser.parse_args()
    return args


def generate_linear_map(orbit_num):
    mymap = {}
    for i in range(orbit_num-1):
        mymap[str(i)] = [i+1]
    return mymap

# {0: [1, 2, 3, 4, 5], 1: [0, 2, 3, 4, 5], 2: [0, 1, 3, 4, 5], 3: [0, 1, 2, 4, 5], 4: [0, 1, 2, 3, 5], 5: [0, 1, 2, 3, 4]}
def generate_all_map(orbital_num):
    mymap = {}
    for i in range(orbit_num):
        all = list(range(orbit_num))
        all.remove(i)
        mymap[str(i)] = all
    return mymap


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
    orbitnums = {
        'H2': 4,
        'LiH': 12
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



    qischem_dict['optimizer']['name'] = args.optimizer
    qischem_dict['variational_form']['depth'] = depths[args.molecule]
    qischem_dict['optimizer']['maxiter'] = evalnums[args.molecule]

    if args.eval_number != -1:
        qischem_dict['optimizer']['maxiter'] = args.eval_number

    orbit_num = orbitnums[args.molecule]

    if args.orbital_reduction == 1:
        if args.molecule == 'LiH':

            orbit_num = int(orbit_num/2) # thanks to the core freezing and orbital reduction

            # extra reduction:
            qischem_dict['operator']['qubit_mapping'] = 'parity'
            orbit_num = orbit_num - 2 # extra orbital reduction thanks to the parity map



            qischem_dict['operator']['freeze_core'] = True
            qischem_dict['operator']['orbital_reduction'] = [-3,-2]
        elif args.molecule == 'H2': # no, we cannot reduce for H2
            pass


    if args.map == "linear":
        gmap = generate_linear_map(orbit_num)
    elif args.map == 'all':
        gmap = generate_all_map(orbit_num)


    qischem_dict['variational_form']['entangler_map'] = gmap


    molecule = molecule_templates[args.molecule]
    qischem_dict['pyscf']['atom'] = molecule  # temporarily set, will be overwritten

    start = starts[args.molecule]
    by    = bys[args.molecule] # How much to increase distance by



    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(qischem_dict)


    #print('\b\b{:2d}'.format(i), end='', flush=True)
    d = args.distance
    qischem_dict['pyscf']['atom'] = molecule.format(d/2)
    solver = QISChem()
    result = solver.run(qischem_dict)
    print(d, result['energy'], result['total_dipole_moment'])

    # the output will be appended to a file
    with open('./singlepoint_' + args.molecule +  '_'+ str(args.distance), 'a') as f:
        f.write("\ndistance: " + str(d) +"\n")
        f.write("energy:" + str(result['energy'])+"\n")
        f.write("dipole moment:" + str(result['total_dipole_moment'])+"\n")
        #f.write("energy:" + str(result['energy']))
