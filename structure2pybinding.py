import pybinding as pb
from ase.io import read
import numpy as np
import scipy as sp
import os

def read_slako(slako_path, atom_types):
    """Reads the on-site and off-diagonal enements from slater koster files. Takes only the pp-pi interactions.

    Args:
        slako_path (str): Path to slater-koster files.
        atom_types (list): List of atom types in string format.

    Returns:
        onsite (dict): onsite energies for different elements
        offdiag (dict): off-diagonal elements for different atom-atom combinations and distances

    """
    # all filenames in list, find filename where all combinations of atom_types are in
    
    # get list of all slakofiles
    slako_files_all = []
    for root, dirs, files in os.walk(slako_path):
        for file in files:
            if file.endswith('.skf'):
                slako_files_all.append(file)

    # get dict with key=atom-combination and value=slako-filename
    slako_files = dict()
    for at1 in atom_types:
        for at2 in atom_types:
            ss = [s for s in slako_files_all if at1 in s[0] if at2 in s[2]]
            slako_files[at1+at2] = ss[0]
    

    # go through all possible atom_type combinations and store onsite and offdiag in dicts. offdiag has as value=[dists, pp-offdiagonalelemets]
    onsite=dict()
    offdiag=dict()
    for at1 in atom_types:
        for at2 in atom_types:
            sfile = os.path.join(slako_path, slako_files[at1+at2])
            gridDist, nGridpoints = np.loadtxt(sfile, skiprows=0, max_rows=1, usecols=(0,1), unpack=True)  # get number of grid-points
            gridDist *= 0.0529177249  # conversion bohr -> nm

            with open(sfile, 'r') as inp:
                lines = inp.readlines()
                
                o = []
                if at1==at2:
                    skip = 3
                else:
                    skip = 2

                for n, line in enumerate(lines):
                    splitline = lines[n+skip].replace(",", " ").strip().split()
                    if n==1 and at1==at2:
                        onsite[at1] = float(line.replace(",", " ").strip().split()[1]) * 27.2114  # Hartree to eV
                    
                    if n==nGridpoints-1:
                        break
                    
                    expanded_line = []
                    for l in splitline:
                        if "*" in l:
                            f,v = l.split("*")
                            for _ in range(0, int(f)):
                                expanded_line.append(float(v))
                        else:
                            expanded_line.append(float(l))
                    o.append(expanded_line[6])  # use only p-p pi interactions



            offdiag[at1+at2] = np.array([np.linspace(0.0, gridDist*(nGridpoints-1), int(nGridpoints)), 
                                            np.array(o) * 27.2114], dtype='object')  # use only p-p pi interactions # Hartree to eV
                                            

    return onsite, offdiag


def create_lattice_pb(structure, onsite, offdiag, cutoff=0.16):
    """Creates a pybinding lattice object based on a ase structure object and onsite and offdiagonal elements.

    Args:
        structure (ase atoms object): ASE atoms object of the structure
        onsite (dict): onsite energies for different elements
        offdiag (dict): off-diagonal elements for different atom-atom combinations and distances
        cutoff (float, optional): cutoff radius for interactions. Small radius reduces Hamiltonian size. Defaults to 0.16 nm.

    Returns:
        pybinding lattice object
    """

    dists_pbc = structure.get_all_distances(mic=True) 

    a1x = structure.get_cell()[0,0] / 10.  # Angstrom to nm
    a1y = structure.get_cell()[0,1] / 10.  # Angstrom to nm
    a2x = structure.get_cell()[1,0] / 10.  # Angstrom to nm
    a2y = structure.get_cell()[1,1] / 10.  # Angstrom to nm

    pb_lattice = pb.Lattice(a1=[a1x, a1y], a2=[a2x, a2y])

    for n, atom in enumerate(structure):
        name = atom.symbol + str(n)
        position = np.array([atom.position[0], atom.position[1]]) / 10.0  # Angstrom to nm  
        onsiteE = onsite[atom.symbol]

        pb_lattice.add_sublattices((name, position, onsiteE))

    m_list = []  # list to check that no combination appears twice
    for m, atom1 in enumerate(structure):
        m_list.append(m)
        for n, atom2 in enumerate(structure):
            if n in m_list:  # no atom combination twice
                continue
            if m==n:  # no onsite
                continue
            d = dists_pbc[m][n] / 10.  # Angstrom to nm
            
            # find closest gridpoint in slakos
            ds, odgs = offdiag[atom1.symbol+atom2.symbol]

            if d > cutoff:
                continue  # don't add this to lattice
            elif d > max(ds):
                continue  # don't add this to lattice
            else:
                ds = abs(ds - d)
                ds, odgs = zip(*sorted(zip(ds, odgs)))
                hopping = odgs[0] 


            name1 = atom1.symbol + str(atom1.index)
            name2 = atom2.symbol + str(atom2.index)
            
            # solve equation Ax=b for x, which is the decomposition of the lattice vectors. b is vector between two atoms
            b = (atom2.position - atom1.position)[:2].T / 10  # Angstrom to nm

            a1 = np.array([a1x, a1y])
            a2 = np.array([a2x, a2y])
            A = np.array([a1, a2]).T

            x = sp.linalg.solve(A, b)

            # use decompsition vector x to determine the hoppings across pbc
            if abs(x[0]) < 0.5 and abs(x[1]) < 0.5:
                pb_lattice.add_one_hopping([0, 0], name1, name2, hopping)
            
            if x[0] < -0.5 and abs(x[1]) < 0.5:
                pb_lattice.add_one_hopping([1, 0], name1, name2, hopping)
            
            if x[0] > 0.5 and abs(x[1]) < 0.5:
                pb_lattice.add_one_hopping([-1, 0], name1, name2, hopping)
                
            if abs(x[0]) < 0.5 and x[1] < -0.5:
                pb_lattice.add_one_hopping([0, 1], name1, name2, hopping)
            
            if abs(x[0]) < 0.5 and x[1] > 0.5:
                pb_lattice.add_one_hopping([0, -1], name1, name2, hopping)
                
            if x[0] < -0.5 and x[1] < -0.5:
                pb_lattice.add_one_hopping([1, 1], name1, name2, hopping)
                
            if x[0] > 0.5 and x[1] > 0.5:
                pb_lattice.add_one_hopping([-1, -1], name1, name2, hopping)
            
    return pb_lattice


def structure2pybinding(structure, slako_path):

    del structure[[atom.index for atom in structure if atom.symbol=='H']]  # deleting H-atoms

    atom_types = [atom.symbol for atom in structure]
    atom_types = list(set(atom_types))  # listing all atom_types

    onsite, offdiag = read_slako(slako_path, atom_types)

    pb_lattice = create_lattice_pb(structure, onsite, offdiag)


    return pb_lattice