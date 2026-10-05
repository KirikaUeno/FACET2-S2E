"""Create bunches (Gaussian and theory-matched).

Moved from beamFunctions.py.
"""

from scipy.stats import moment, gennorm
from scipy.special import gamma
import numpy as np
from pmd_beamphysics import ParticleGroup

from .manipulation import cut_length


## Bunch support functions

### Create a bunch

def make_simple_bunch(file, n = 0, save_path = ''):
    """Create a Gaussian bunch with statistics matched to an input beam.

    The transverse and momentum distributions are drawn from normal
    distributions matched to the input beam rms values.
    The transverse emittance is increased (alpha = 0, rms are the same);
    The longitudinal emittance is kept approximately the same (alpha = 0, rms by pz is taken from a 0.1um slice).

    Parameters:
        file: Base path of input beam file without the .h5 extension.
        n: Number of macro particles to generate. If 0, the input beam size is used.
        save_path: Output file path without extension. If "", file+'_simple.h5' is used.

    Returns:
        ParticleGroup: Generated bunch.
    """
    beam = ParticleGroup(file + '.h5')
    
    N = np.size(beam.x)
    N = N if n==0 else n
    charge = beam.charge
    data = {'x': np.zeros(N), 'px': np.zeros(N), 'y': np.zeros(N), 'py': np.zeros(N), 'z': np.zeros(N), 'pz': np.zeros(N), 't': np.zeros(N), 'status': np.ones(N).astype(int), 'weight': np.ones(N)*charge/N, 'species': 'electron', 'id': np.arange(N).astype(int)}
    P1 = ParticleGroup(data = data)
    
    Pslice = cut_length(beam, length = 1e-7)
    
    P1.x = np.random.normal(0, 1*moment(beam.x, moment=2) ** 0.5, N)
    P1.y = np.random.normal(0, 1*moment(beam.y, moment=2) ** 0.5, N)
    P1.px = np.random.normal(0, 1*moment(beam.px, moment=2) ** 0.5, N)
    P1.py = np.random.normal(0, 1*moment(beam.py, moment=2) ** 0.5, N)
    P1.pz = np.random.normal(np.mean(beam.pz), 1*moment(Pslice.pz, moment=2) ** 0.5, N)
    P1.t = np.random.normal(0, 1*moment(beam.t, moment=2) ** 0.5, N)
    
    match_impact_file = file + '_simple' + '.h5' if save_path=='' else save_path + '.h5'
    P1.write(match_impact_file)
    return P1


def make_simple_bunch_flatter(file, n = 0, save_path = ''):
    """Create a flatter bunch using a generalized normal time distribution.

    Similar to make_simple_bunch, but the longitudinal time coordinate is
    sampled from a generalized normal distribution to produce a flatter
    longitudinal profile.

    Parameters:
        file: Base path of input beam file without the .h5 extension.
        n: Number of macro particles to generate. If 0, the input beam size is used.
        save_path: Output file path without extension. If empty, file+'_simple.h5' is used.
    """
    beam = ParticleGroup(file + '.h5')
    
    N = np.size(beam.x)
    N = N if n==0 else n
    charge = beam.charge
    data = {'x': np.zeros(N), 'px': np.zeros(N), 'y': np.zeros(N), 'py': np.zeros(N), 'z': np.zeros(N), 'pz': np.zeros(N), 't': np.zeros(N), 'status': np.ones(N).astype(int), 'weight': np.ones(N)*charge/N, 'species': 'electron', 'id': np.arange(N).astype(int)}
    P1 = ParticleGroup(data = data)
    
    Pslice = cut_length(beam, length = 1e-7)
    
    P1.x = np.random.normal(0, 1*moment(beam.x, moment=2) ** 0.5, N)
    P1.y = np.random.normal(0, 1*moment(beam.y, moment=2) ** 0.5, N)
    P1.px = np.random.normal(0, 1*moment(beam.px, moment=2) ** 0.5, N)
    P1.py = np.random.normal(0, 1*moment(beam.py, moment=2) ** 0.5, N)
    P1.pz = np.random.normal(np.mean(beam.pz), 1*moment(Pslice.pz, moment=2) ** 0.5, N)
    P1.t = gennorm.rvs(4, size=N)*((moment(beam.t, moment=2)/(gamma(3/4)/gamma(1/4))) ** 0.5)
    
    match_impact_file = file + '_simple' + '.h5' if save_path=='' else save_path + '.h5'
    P1.write(match_impact_file)
    return P1


def make_simple_bunch_standalone(N = 0, meanPzMeV = 125 , moments=[0.3e-3, 0.2e-3, 0.4e-3, 0.2e-3, 0.58e-3, 0], charge = 1e-9, save_path = '', means=[0,0,0,0,0,0]):
    """Create a Gaussian bunch from explicit statistical moments.

    Parameters:
        N: Number of macro particles.
        meanPzMeV: Mean longitudinal momentum in MeV/c.
        moments: RMS values in x, xp, y, yp, z, pz.
        charge: bunch charge.
        save_path: Output file path without extension.
        means: Mean values for x, xp, y, yp, z, pz.

    Returns:
        ParticleGroup: Generated bunch.
    """

    N = int(N)
    data = {'x': np.zeros(N), 'px': np.zeros(N), 'y': np.zeros(N), 'py': np.zeros(N), 'z': np.zeros(N), 'pz': np.zeros(N), 't': np.zeros(N), 'status': np.ones(N).astype(int), 'weight': np.ones(N)*charge/N, 'species': 'electron', 'id': np.arange(N).astype(int)}
    P1 = ParticleGroup(data = data)
    
    P1.x = np.random.normal(means[0], moments[0], N)
    P1.px = np.random.normal(means[1], moments[1]*meanPzMeV*1e6, N)
    P1.y = np.random.normal(means[2], moments[2], N)
    P1.py = np.random.normal(means[3], moments[3]*meanPzMeV*1e6, N)
    P1.t = np.random.normal(means[4], moments[4], N)/3e8
    P1.pz = np.random.normal(meanPzMeV*1e6, moments[5], N)
    
    match_impact_file = save_path + '.h5'
    P1.write(match_impact_file)
    return P1


def make_simple_bunch_theory_from_bunch_sims(bunch_file, mean_lattice_P0C_MeV, means_shift=[0,0,0,0,0,0]):
    """Convert a BMAD bunch into theory coordinates for map-based calculations.

    Parameters:
        bunch_file: Base path of the bunch file without extension.
        mean_lattice_P0C_MeV: Reference lattice momentum in MeV/c (the bunch will have this <pz>).
        means_shift: Shifts to apply to x, xp, y, yp, z, delta.

    Returns:
        np.ndarray: Nx6 array in the order [x, xp, y, yp, z, delta].
    """
    sim_bunch = ParticleGroup(bunch_file+".h5")
    return np.stack((sim_bunch.x+means_shift[0], sim_bunch.xp+means_shift[1], sim_bunch.y+means_shift[2], sim_bunch.yp+means_shift[3], -3e8*sim_bunch.t, (sim_bunch.pz*1e-6-mean_lattice_P0C_MeV)/mean_lattice_P0C_MeV), axis=1)
