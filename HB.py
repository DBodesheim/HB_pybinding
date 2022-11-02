from dataclasses import asdict
import numpy as np
import pybinding as pb
from pybinding.constants import phi0
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import copy
import matplotlib as mpl
import pickle

def save_pickle(obj, filename):
    """Saves a pickle of an object to file.

    Args:
        obj : Object to pickle.
        filename (str): Filename to save to.
    """
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

def load_pickle(filename):
    """Loads pickle of an object from file.

    Args:
        filename (str): Filename load from.

    Returns:
        object
    """
    with open(filename, 'rb') as input_file:
        obj = pickle.load(input_file)
    return obj

class Butterfly():
    """
    Class for the calculation of the butterfly spectrum based on tight binding.
    
    Input:
        pb_lattice: pybinding lattice object
        params (dict): parameter dictionary. If not specified, some default parameters are used.
    """


    default_params = {
            'emin': -10,  # energy minimum [eV]
            'emax': 10,  # energy maximum [eV]
            'enum': 1000,  # number of energy steps
            'Bmin': 0,  # B minimum [T]
            'Bmax': 240000,  # B maximum [T]
            'Bnum': 1000,  # number of B steps
            'num_cores': 1,  # number of cores to use
            'a1': 50,  # supercell-size a1xa2
            'a2': 50,  # supercell-size a1xa2
            'broadening': 0.0005,  # broadening for kpm method
            'num_random': 3  # number of random vectors for kpm method
         }
    def __init__(self, pb_lattice, params=None):

        self.pb_lattice = pb_lattice
        if params==None:
            print('No parameters specified. Defualt parameters will be used.')
            self.params = default_params
        else:
            self.params = params

        if self.params['emin'] > self.params['emax']:
            print('WARNING: EMIN LARGER THAN EMAX')
        if self.params['Bmin'] > self.params['Bmax']:
            print('WARNING: BMIN LARGER THAN BMAX')
        if self.params['broadening']=='min':
            self.params['broadening'] = abs(self.params['emin'] - self.params['emax'])/self.params['enum']
        elif self.params['broadening'] < abs(self.params['emin'] - self.params['emax'])/self.params['enum']:
            print('Broadening is {} which is smaller than the set energy resolution ({}).\n'.format(self.params['broadening'], abs(self.params['emin'] - self.params['emax'])/self.params['enum']))
            print('Setting broadening to {}'.format(abs(self.params['emin'] - self.params['emax'])/self.params['enum']))
            self.params['broadening'] = abs(self.params['emin'] - self.params['emax'])/self.params['enum']

        self.energies = np.linspace(self.params['emin'], self.params['emax'], self.params['enum'])
        self.Bs = np.linspace(self.params['Bmin'], self.params['Bmax'], self.params['Bnum'])


        
    def get_Eshift(self):
        """Calculate Energy shift to center the data around 0 eV.
        Uses max and min value of butterfly spectrum that are not NaN and calculates
        shift to center spectrum around 0 eV.
        Probably only makes sense if one obtained whole energy spectrum.
        Mainly important for plotting purposes.

        Raises:
            AssertionError: If no data was calculated yet.
        Returns:
            float: Energy shift
        """
        if hasattr(self, 'data'):
            
            foundmax=False
            foundmin=False
            emax_new = 0
            emin_new = 0
            for e, d in zip(self.energies, np.isnan(self.data.T)):
                if False in d:
                    emax_new = e
                    if foundmax and not foundmin:
                        emin_new = e
                        foundmin=True
                else:
                    foundmax=True
            dy = abs((emax_new-emin_new))
            Eshift = dy/2 - emax_new
            self.Eshift = Eshift
            return self.Eshift       
        else:
            raise AssertionError('No data available. First run create_butterfly routine first or load data.')
            
    def set_Eshift(self, Eshift):
        """Set Energy shift for shifting Energy scale [eV]. 
        Mainly important for plotting purposes.

        Args:
            Eshift (float): Energy shift in eV.
        """

        self.Eshift = Eshift

    def create_butterfly(self):
        """Calculates butterfly spectrum

        Returns:
            numpy array: butterfly spectrum consisting of an array DOS for different magnetic fields.
        """
        if self.params['num_cores'] > 1:
            try:
                from joblib import Parallel, delayed
            except ImportError:
                print('joblib package not installed. Changing to serial calculations (num_cores=1)')
                self.params['num_cores'] = 1

        def constant_magnetic_field(B):
            @pb.hopping_energy_modifier
            def function(energy, x1, y1, x2, y2):
                y = 0.5 * (y1 + y2)                # the midpoint between two sites
                y *= 1e-9                          # scale from nanometers to meters
                A_x = B * y                        # vector potential along the x-axis

                peierls = A_x * (x1 - x2)          # integral of (A * dl) from position 1 to position 2
                peierls *= 1e-9            
                
                return energy * np.exp(1j * 2*np.pi/phi0 * peierls) # the Peierls substitution
            return function
        
        def DOS_B(B):
            model = pb.Model(self.pb_lattice, 
                            pb.primitive(a1=self.params['a1'], a2=self.params['a2']), 
                            constant_magnetic_field(B=B))
            kpm = pb.kpm(model, silent=True)
            
            dos = kpm.calc_dos(energy=self.energies, broadening=self.params['broadening'], num_random=self.params['num_random'])
            return dos.data

        if self.params['num_cores']==1 or self.params['num_cores']==None:
            print('Number of cores: 1. Running in serial')
            butterflydata = []
            for B in self.Bs:
                butterflydata.append(DOS_B(B))

        elif self.params['num_cores'] > 1:
            butterflydata = Parallel(n_jobs=self.params['num_cores'], verbose=10)(delayed(DOS_B)(B) for B in self.Bs)

        self.data = np.array(butterflydata)


    def plot(self, 
                vmax=1.0, vmin=0.001, 
                fig=None, ax=None,
                ylabel=True, xlabel=True,
                kT=False, 
                color_map='Purples', 
                cbar=False):
    

        if kT:
            Bmin = self.params['Bmin'] / 1000  # T to kT
            Bmax = self.params['Bmax'] / 1000  # T to kT
        else:
            Bmin = self.params['Bmin']
            Bmax = self.params['Bmax']
        
        if ax==None and fig==None:
            fig, ax = plt.subplots(figsize=(8,6))
        
        cmap = copy.copy(mpl.cm.get_cmap(color_map))
        
        cmap.set_bad(cmap(0))  # for nan, value of 0 is assumed
        
        data = self.data.copy()

        data[data==-np.inf] = np.nan  # remove -np.inf
        data[data==np.inf] = np.nan  # remove np.inf
        data -= np.nanmin(data)  # shifting data so that min value is 0 
        data_normalized = data/np.nanmax(data)  # normalize data
        
        
        if hasattr(self, 'Eshift'):
            print('Using Eshift={} to shift energy scale.'.format(self.Eshift))
            Eshift = self.Eshift
        else:
            print('No Eshift defined. Using Eshift=0.')
            Eshift = 0

            
        ext = [Bmin, Bmax, self.params['emin']+Eshift, self.params['emax']+Eshift]
        im = ax.imshow(data_normalized.T, cmap=cmap, 
                extent=ext, aspect="auto", origin='lower', 
                norm=LogNorm(vmin, vmax),  # plot on a log scale
                    )
        
        # setting x- and y-label
        if ylabel==True:
            ax.set_ylabel('Energy [eV]')
        if xlabel==True:
            if kT:
                ax.set_xlabel('B [kT]')
            else:
                ax.set_xlabel('B [T]')
        
        # setting x- and y-limits
        ax.set_xlim([Bmin, Bmax])
        ax.set_ylim([self.params['emin']+Eshift, self.params['emax']+Eshift])

        # use colorbar
        if cbar==True:
            cbar = plt.colorbar(im)
            cbar.ax.set_ylabel('Normalized DOS')
                
        plt.tight_layout()
        
        return fig, ax