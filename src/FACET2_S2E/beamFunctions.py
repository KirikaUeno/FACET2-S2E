import math
from scipy.stats import moment
from scipy.stats import gennorm
from scipy.special import gamma
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
import pprint
from copy import copy
import matplotlib.pyplot as plt

#import mplstyle
from matplotlib.ticker import AutoMinorLocator
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from Experimental_functions import *

# from .UTILITY_quickstart import (
#     initializeTao,
#     trackBeam,
#     getBeamAtElement,
# )

from .UTILITY_quickstart import *

from .UTILITY_linacPhaseAndAmplitude import matchStringWrapper

"""Beam preparation utilities for FACET2-S2E simulation workflows.

This module contains functions to create bunches (Gaussian and
theory-matched), and to modify an existing bunch as a whole (sizes,
means, chirps, correlations) or cut a certain length out of it.

Split out of the former functionsForSims.py.
"""

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


## Modify the bunch as a whole (sizes, means, chirps, correlations)

def modifyInputBeamSimple(inputBeamFilePath, numMacroParticles = None, timeCenterTF = True):
    """Prepare an input beam for Tao by optionally downsampling and centering. Almost the same as Nathans', but without Twiss matching.

    The beam is drift_to_z(), set z=0, and optionally time-centered.
    If numMacroParticles is provided, the beam is randomly subsampled and weights are adjusted.

    Parameters:
        inputBeamFilePath: Path to the input beam file, including extension.
        numMacroParticles: Target number of macroparticles.
        timeCenterTF: If True, subtract the mean time to center the bunch and avoid cavity phase mismatch.

    Returns:
        ParticleGroup: Modified beam ready for use with Tao.
    """
    P = ParticleGroup(inputBeamFilePath)

    if numMacroParticles:
        if numMacroParticles>0:
            initialImportSize = np.size(P.x)
            numMacroParticles = int(numMacroParticles)
            P = P[random.sample(range(initialImportSize), numMacroParticles)]
            P.weight = P.weight * (initialImportSize / numMacroParticles)

    P.drift_to_z()
    P.z = np.zeros(np.size(P.x))
    # Time center
    if timeCenterTF:
        P.t=P.t-np.mean(P.t) # This is OK because present beam doesn't have different weights; np.unique(P.weight)
        
    return P

def sqrtm_psd(M, tol=1e-14):
    """Compute the positive-semidefinite square root of a symmetric matrix.
    
    Returns:
        tuple: (matrix_sqrt, eigenvalues)
    """
    w, V = np.linalg.eigh(M)
    w_clipped = np.clip(w, 0.0, None)  # allow zero
    return V @ np.diag(np.sqrt(w_clipped)) @ V.T, w

def invsqrtm_psd(M, tol=1e-14):
    """Compute the inverse square root of a symmetric matrix, with small eigenvalues treated as zero.
    
    Returns:
        tuple: (matrix_sqrt, eigenvalues)
    """
    w, V = np.linalg.eigh(M)
    w_inv = np.zeros_like(w)
    mask = w > tol
    w_inv[mask] = 1.0 / np.sqrt(w[mask])
    # zero eigenvalues remain zero → projection
    return V @ np.diag(w_inv) @ V.T, w


def edit_bunch_parameters_from_PG(P_arg, pzMeV=None, moments=[None, None, None, None, None, None], correlations=[None,None,None], means=[0,0,0,0,0],
                                  charge=-1, betaX=None, alphaX=None, emittanceX=None, betaY=None, alphaY=None, emittanceY=None, path_to_write='temp_beam/temp_e'):
    '''
    moments are for x, xp, y, yp, z, pz
    means are for x, xp, y, yp, z
    xp and yp are in radians
    moments are RMS sizes

    if betaX and alphaX are supplied, the X phase space will have the emittance corresponding to moments[0] (size x = sqrt{epsilon beta}), and moments[1] will be overwritten. Same for Y.
    '''
    P = P_arg.copy()
    
    if correlations[0] is None:
        correlations[0] = np.mean((P.x - np.mean(P.x))*(P.xp - np.mean(P.xp)))/np.std(P.x - np.mean(P.x))*np.std(P.xp - np.mean(P.xp)) if (np.std(P.xp - np.mean(P.xp))!=0 and np.std(P.x - np.mean(P.x))!=0) else 0
    if correlations[1] is None:
        correlations[1] = np.mean((P.y - np.mean(P.y))*(P.yp - np.mean(P.yp)))/np.std(P.y - np.mean(P.y))*np.std(P.yp - np.mean(P.yp)) if (np.std(P.yp - np.mean(P.yp))!=0 and np.std(P.y - np.mean(P.y))!=0) else 0
    if correlations[2] is None:
        correlations[2] = np.mean((P.pz - np.mean(P.pz))*(P.t - np.mean(P.t))*3e8)/np.std(P.pz - np.mean(P.pz))*np.std((P.t - np.mean(P.t))*3e8) if (np.std((P.t - np.mean(P.t))*3e8)!=0 and np.std(P.pz - np.mean(P.pz))!=0) else 0
    
    sigmaMatrixX=np.array([[-1,-1],[-1,-1]], dtype=float)
    sigmaMatrixY=np.array([[-1,-1],[-1,-1]], dtype=float)
    sigmaMatrixZ=np.array([[-1,-1],[-1,-1]], dtype=float)
    sigmaMatrixX[0][0] = -1 if moments[0]==None else moments[0]**2
    sigmaMatrixX[1][1] = -1 if moments[1]==None else moments[1]**2
    sigmaMatrixY[0][0] = -1 if moments[2]==None else moments[2]**2
    sigmaMatrixY[1][1] = -1 if moments[3]==None else moments[3]**2
    sigmaMatrixZ[0][0] = -1 if moments[4]==None else moments[4]**2
    sigmaMatrixZ[1][1] = -1 if moments[5]==None else moments[5]**2
    
    # charge
    if charge!=-1:
        P.charge = charge
    N = len(P.x)
    
    # longitudinal momentum (pz), t and z
    if pzMeV is not None and pzMeV>0:
        P.pz = P.pz*1e6*pzMeV/np.mean(P.pz)

    meanpz = np.mean(P.pz)
    P.pz = P.pz - meanpz
    means[4] = means[4] if means[4]!=None else np.mean(P.t)*3e8
    P.t = P.t - np.mean(P.t)

    if not ((sigmaMatrixZ[1,1]==-1) and (sigmaMatrixZ[0,0]==-1)):
        if sigmaMatrixZ[0][0]==-1:
            sigmaMatrixZ[0][0] = np.std(P.t*3e8)**2
        if sigmaMatrixZ[1][1]==-1:
            sigmaMatrixZ[1][1] = np.std(P.pz)**2
        sigmaMatrixZ[0][1] = correlations[2]*np.sqrt(sigmaMatrixZ[0][0]*sigmaMatrixZ[1][1])
        sigmaMatrixZ[1][0] = sigmaMatrixZ[0][1]
        if np.std(P.t)==0 or np.std(P.pz)==0:
            P.t = np.random.normal(0, 1, N)/3e8
            P.pz = np.random.normal(0, 1, N)
        Z = np.vstack((-P.t*3e8, P.pz))
        Sigma_current = np.cov(Z, bias=True)
        S_target_sqrt, _ = sqrtm_psd(sigmaMatrixZ)
        S_current_invsqrt, _ = invsqrtm_psd(Sigma_current)
        T = S_target_sqrt @ S_current_invsqrt
        Z_new = T @ Z
        P.t = -Z_new[0]/3e8
        P.pz = Z_new[1]

    P.pz = P.pz + meanpz
    P.t = P.t + means[4]/3e8
            
    # means before
    means[0] = means[0] if means[0]!=None else np.mean(P.x)
    means[1] = means[1] if means[1]!=None else np.mean(P.xp)
    means[2] = means[2] if means[2]!=None else np.mean(P.y)
    means[3] = means[3] if means[3]!=None else np.mean(P.yp)
    P.x = P.x - np.mean(P.x)
    P.px = P.px - np.mean(P.px)
    P.y = P.y - np.mean(P.y)
    P.py = P.py - np.mean(P.py)

    #Apply linear matching X
    if (betaX is not None) and (alphaX is not None):
        current_sqrt_emmitance = np.sqrt(np.sqrt( np.mean(P.x**2)*np.mean(P.xp**2) - np.mean(P.x*P.xp)**2 ))
        if current_sqrt_emmitance==0:
            print("edit_bunch_parameters_from_PG: Zero emittance in X. Filling with Gaussian with the target emittance.")
            P.x = np.random.normal(0, np.sqrt(emittanceX), N)
            P.px = np.random.normal(0, np.mean(P.pz)*np.sqrt(emittanceX), N)
        P.twiss_match(plane='x', beta = betaX, alpha = alphaX, inplace=True)
        if current_sqrt_emmitance!=0:
            if emittanceX is not None:
                P.x = P.x*np.sqrt(emittanceX)/current_sqrt_emmitance
                P.px = P.px*np.sqrt(emittanceX)/current_sqrt_emmitance
            
    #if doing second moments matrix instead of beta alpha emittance
    else:
        if not ((sigmaMatrixX[1,1]==-1) and (sigmaMatrixX[0,0]==-1)):
            if sigmaMatrixX[0][0]==-1:
                sigmaMatrixX[0][0] = np.std(P.x)**2
            if sigmaMatrixX[1][1]==-1:
                sigmaMatrixX[1][1] = np.std(P.xp)**2
            sigmaMatrixX[0][1] = correlations[0]*np.sqrt(sigmaMatrixX[0][0]*sigmaMatrixX[1][1])
            sigmaMatrixX[1][0] = sigmaMatrixX[0][1]
            if np.std(P.x)==0 or np.std(P.xp)==0:
                P.x = np.random.normal(0, 1, N)
                P.px = np.random.normal(0, np.mean(P.pz)*1, N)
            X = np.vstack((P.x, P.xp))
            Sigma_current = np.cov(X, bias=True)
            S_target_sqrt, _ = sqrtm_psd(sigmaMatrixX)
            S_current_invsqrt, _ = invsqrtm_psd(Sigma_current)
            T = S_target_sqrt @ S_current_invsqrt
            X_new = T @ X
            P.x = X_new[0]
            P.px = X_new[1]*np.mean(P.pz)

    if (betaY is not None) and (alphaY is not None):
        current_sqrt_emmitance = np.sqrt(np.sqrt( np.mean(P.y**2)*np.mean(P.yp**2) - np.mean(P.y*P.yp)**2 ))
        if current_sqrt_emmitance==0:
            print("edit_bunch_parameters_from_PG: Zero emittance in Y. Filling with Gaussian with the target emittance.")
            P.y = np.random.normal(0, np.sqrt(emittanceY), N)
            P.py = np.random.normal(0, np.mean(P.pz)*np.sqrt(emittanceY), N)
        P.twiss_match(plane='y', beta = betaY, alpha = alphaY, inplace=True)
        if current_sqrt_emmitance!=0:
            if emittanceY is not None:
                P.y = P.y*np.sqrt(emittanceY)/current_sqrt_emmitance
                P.py = P.py*np.sqrt(emittanceY)/current_sqrt_emmitance
    else:
        if not ((sigmaMatrixY[1,1]==-1) and (sigmaMatrixY[0,0]==-1)):
            if sigmaMatrixY[0][0]==-1:
                sigmaMatrixY[0][0] = np.std(P.y)**2
            if sigmaMatrixY[1][1]==-1:
                sigmaMatrixY[1][1] = np.std(P.yp)**2
            sigmaMatrixY[0][1] = correlations[1]*np.sqrt(sigmaMatrixY[0][0]*sigmaMatrixY[1][1])
            sigmaMatrixY[1][0] = sigmaMatrixY[0][1]
            if np.std(P.y)==0 or np.std(P.yp)==0:
                P.y = np.random.normal(0, 1, N)
                P.py = np.random.normal(0, np.mean(P.pz)*1, N)
            Y = np.vstack((P.y, P.yp))
            Sigma_current = np.cov(Y, bias=True)
            S_target_sqrt, _ = sqrtm_psd(sigmaMatrixY)
            S_current_invsqrt, _ = invsqrtm_psd(Sigma_current)
            T = S_target_sqrt @ S_current_invsqrt
            Y_new = T @ Y
            P.y = Y_new[0]
            P.py = Y_new[1]*np.mean(P.pz)

    # means after
    P.x = P.x + means[0]
    P.px = P.px + means[1]*np.mean(P.pz)
    P.y = P.y + means[2]
    P.py = P.py + means[3]*np.mean(P.pz)

    P.write(path_to_write+".h5")
    return P

def edit_bunch_parameters(file_ext, pzMeV=None, moments=[None,None,None,None,None,None], correlations=[None,None,None], means=[0,0,0,0,0], charge=-1,
                          betaX=None, alphaX=None, emittanceX=None, betaY=None, alphaY=None, emittanceY=None, path_to_write='temp_beam/temp_e'):
    '''Edit the bunch parameters.
    file_ext: Base file path without the .h5 extension.

    moments are for x, xp, y, yp, z, pz
    means are for x, xp, y, yp, z
    xp and yp are in radians
    moments are RMS sizes

    if betaX and alphaX are supplied, the X phase space will have the emittance corresponding to moments[0] (size x = sqrt{epsilon beta}), and moments[1] will be overwritten. Same for Y.
    '''
    return edit_bunch_parameters_from_PG(ParticleGroup(file_ext + ".h5"), pzMeV=pzMeV, moments=moments, correlations=correlations, means=means, charge=charge,
                                  betaX=betaX, alphaX=alphaX, emittanceX=emittanceX, betaY=betaY, alphaY=alphaY, emittanceY=emittanceY, path_to_write=path_to_write)

def cut_length(particle_group, length = 0, drift_to_z = True):
    """Return a slice of the beam around its mean arrival time.

    Parameters:
        particle_group: Input ParticleGroup.
        length: Full longitudinal window in meters.
        drift_to_z: If False, the beam will be drift_to_t to <t> after the slicing.

    Returns:
        ParticleGroup: Sliced beam.
    """

    P = particle_group.copy()
    P.drift_to_z()
    indexes_to_leave = []
    meanT = np.mean(P.t)
    for (i,p) in enumerate(P):
        if(np.abs(p.t-meanT)<(length/(3e8))):
            indexes_to_leave.append(i)
    indices = np.array(indexes_to_leave)
    Ptemp = P[indices]
    if not drift_to_z:
        Ptemp.drift_to_t()
    return Ptemp
