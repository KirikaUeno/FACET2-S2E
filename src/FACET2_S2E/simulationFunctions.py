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

"""Simulation initialization/run/tuning and scan utilities for FACET2-S2E.

This module contains the high-end functions to build a Tao object from
an experiment (or a synthetic bunch), set the beam, run a tracked
simulation, tune the lattice to match desired cavity/beam energies,
deal with dipole fields, run 1D scans, and read lattice/map parameters
(R-matrix, Taylor map elements, sextupole settings).

Split out of the former functionsForSims.py.
"""

from .beamFunctions import (
    make_simple_bunch_standalone,
    make_simple_bunch_theory_from_bunch_sims,
    modifyInputBeamSimple,
    edit_bunch_parameters_from_PG,
    edit_bunch_parameters,
)
from .DAQdatasetToSimFunctions import (
    edit_tao_based_on_experiment_database,
    edit_energy_tao_based_on_experiment_database,
    save_dipole_energies_from_the_DAQ_database,
)
from .plottingFunctions import make_a_plot

## Initialize and run a simulation


### High end

def get_tao_from_experiment(experiment="", scan_number="", date="", start='L0AFEND', finish='PR11375', filepath="/sdf/group/facet/kladov/FACET2_S2E", locationsToSave = [],
                            csrTF=False, lscTF=False, file_ext = "", energy=None, N_in_simple_bunch=5e4, N_to_use_from_file=None, tune_dipoles_to_125_335_4500_10000_MeV=False, tune_dipoles=False,
                            correctors_coef=0, correctors_from_beg=False, run=True, gaussFromExternal=False, edit_only_energy_from_exp=False, energy_edit_on_beam=False, verbose=False,
                            lattice='setLattice_configs/2024-10-22_oneBunch-Copy1.yml', moments=[None,None,None,None,None,None], means=[0,0,0,0,None], charge=1.6e-9, sr_wakes_on=False, lr_wakes_on=False,
                            desired_beam_energies_for_the_feedback=None, desired_P0Cs_MeV=[None,None,None,None], grid_size=[32,32,32], lsc_method="slice", csr_method="1_dim", n_bin=32,
                            edited_bunch_energy_at_checkpoints_MeV=[None, None, None, None], beam_edits=True):
    '''
    experiment: "BEAMPHYS" is an example.
    scan_number: DAQ scan number. "14438" is an example.
    date: the date when the scan was taken in a specific form. "/2026/20260121" is an example.

    start: where the simulation starts. This 1) can affect the initial bunch energy (see "energy"), 2) determines the start for the energy_edit_on_beam and run_initialized_sim functions.
    finish: determines the finish for the energy_edit_on_beam and run_initialized_sim functions.

    filepath: Path to the package. I haven't found a way to do this automatically yet.

    lattice: additional lattice settings to use.

    file_ext: the beam file. Not required (default is ""). If it is "", a Gaussian bunch will be created with "moments". "moments" in this case is required!
    N_to_use_from_file: randomly choose N_to_use_from_file particles from the external bunch.
    gaussFromExternal: if True, a Gaussian bunch will be created instead of the supplied bunch, with sizes being the same as in the supplied bunch.
    alpha in this case is set to 0, and the emittance is changed accordingly.
    N_in_simple_bunch: number of particles in the created Gaussian bunch (even if created from the external file with gaussFromExternal).

    locationsToSave: Usually, tao saves the bunch to RAM (I believe). The beam will be saved to the disk at the locationsToSave entries.
    Note that it will give an error if Tao does not save a provided location to RAM. To add the location to RAM, go to the quickstart -> initializeTao -> edit the "set beam add_saved_at" lines.

    csrTF, lscTF, sr_wakes_on, lr_wakes_on: bool settings for the collective effects. Work separately.
    grid_size: the grid size used by SC or CSR if they are 3d.
    lsc_method: off, fft_3d or slice.
    csr_method: off, steady_state_3d or 1_dim.
    n_bin: number of longitudinal slices in the slice/1_dim methods.

    energy: initial energy of the bunch in MeV (if positive number).
    Set energy=-1 to use the energy from the beam file, or energy=None to use the energy from the lattice / experiment (if experiment and scan_number are provided).

    moments: list of 6 numbers, the RMS sizes of the bunch in x, xp, y, yp, z, pz (in meters, radians, meters, radians, meters, MeV/c).
    Set a moment to -1 to keep the same as in the input file (applicable to each number). Default is -1 for all.

    means: list of 5 numbers, the means of the bunch in x, xp, y, yp, z (in meters, radians, meters, radians, meters).
    Set a mean to -1 to keep the same as in the input file (applicable to each number). Default is [0,0,0,0,-1] (I needed a centered bunch. Change this if needed).
    z and t are set to 0 in the set_beam() by default.
    
    desired_P0Cs_MeV: This settings allows using different cavity energies (from 125, 335, 4500, and 10000) while preserving the linac geometry.
    For example, setting it to [124, None, None, None], will adjust the injector and L1 cavities to have [124, 335, 4500, 10000] MeV. This will result in a non-zero <x> inside of the dogleg.

    energy_edit_on_beam: The beam feedback. If collective effects slow the bunch down, the cavity voltages will be adjusted to match the desired_beam_energies_for_the_feedback.
    desired_beam_energies_for_the_feedback: <pz> in eV that you would like to see before the dogleg, bc11, bc14, and bc20 (see the energy_edit_on_beam function).

    correctors_coef: what corrector strength to use from the DAQ database. -1/10 for the experimental values; 0 to turn them off.
    correctors_from_beg: if True, all DAQ saved correctors will be used. If False, correctors will be enabled from BX0FBEG.

    edit_only_energy_from_exp: if true, the quadrupoles, sextupoles, dipoles, and correctors will not be loaded from the DAQ database.

    tune_dipoles: if True, the dipole fields are adjusted using DB_FIELD to the corresponding angle and rho from the .tao lattice at some energies:
    Will tune to the DAQ values (it saves energy) if experiment and scan_number are provided, and to the default [125, 335, 4500, 10000] otherwise.
    tune_dipoles_to_125_335_4500_10000_MeV: if True, the dipole magnetic fields are set to [125, 335, 4500, 10000] even if the DAQ is provided.
    If False, the simulation is the same as Nathan's, where the dipole strength changes with the lattice energy.
    '''
    tao = initializeTao(filePath = filepath, loadCustomLatticeTF=True, csrTF=csrTF, lscTF=lscTF, latticeFile=lattice, bmad_grid_size=grid_size, verbose=verbose, sr_wakes_on=sr_wakes_on, lr_wakes_on=lr_wakes_on, lsc_method=lsc_method, csr_method=csr_method, n_bin=n_bin, autoLoadActiveFile=False)
    
    dipoleEnergies_MeV = [125, 335, 4500, 10000]
    # copy the experiment data
    if experiment!="" and scan_number!="":
        ds = DATASET("", experiment, scan_number, pathfull = "".join(["/sdf/data/ad/fs/transition/nfs/slac/g/facet/matlab/data_prod/nas-li20-pm00/", experiment, date]))
        # check if magnets data is not needed. Dogleg energy is always 125 MeV (maybe need to change to the mean of 'BEND_IN10_661_BDES' and 'BEND_IN10_751_BDES' (they are in GeV), so that the magnet strengths are actually correct)
        if edit_only_energy_from_exp:
            tao = edit_energy_tao_based_on_experiment_database(tao, ds)
        else:
            tao = edit_tao_based_on_experiment_database(tao, ds, correctors_coef=correctors_coef, correctors_from_beg=correctors_from_beg)
            dipoleEnergies_MeV = save_dipole_energies_from_the_DAQ_database(ds)

    if tune_dipoles_to_125_335_4500_10000_MeV:
        dipoleEnergies_MeV = [125, 335, 4500, 10000]

    fields = save_dipoles(tao, dipoleEnergies_MeV)

    # tao.cmd(f'set global lattice_calc_on = T')
    # deal with the bunch
    current_e_start = tao.ele_gen_attribs(start)["P0C"]*1e-6 if energy is None else energy
    folder = filepath + "/"
    file = folder+ "temp_beam/temp"
    if beam_edits:
        if file_ext=='':
            moments = [0 if moment is None else moment for moment in moments]
            make_simple_bunch_standalone(N = N_in_simple_bunch, meanPzMeV = current_e_start, moments=moments, save_path = file, charge=charge)
        else:
            energy_from_file = None if energy==-1 else current_e_start
            edit_bunch_parameters(file_ext, pzMeV=energy_from_file, moments=moments, means=means, charge=charge, path_to_write=file)
            if gaussFromExternal:
                P = ParticleGroup(file+".h5")
                moments = [np.std(P.x),np.std(P.xp),np.std(P.y),np.std(P.yp),np.std(P.t)*3e8,np.std(P.pz)]
                make_simple_bunch_standalone(N = N_in_simple_bunch, meanPzMeV = current_e_start, moments=moments, save_path = file)
                edit_bunch_parameters(file, pzMeV=current_e_start, moments=moments, means=means, charge=charge, path_to_write=file)
    set_beam(tao, file, numMacroParticles=None if (gaussFromExternal or file_ext=='') else N_to_use_from_file)

    # energy feedback on beam
    if energy_edit_on_beam:
        tao = edit_energy_based_on_beam_all(tao, start, file, verbose=verbose, desired_beam_energies=desired_beam_energies_for_the_feedback, finalnumMacroParticles=N_to_use_from_file)

    if tune_dipoles:
        tao = treat_dipoles(tao, fields)

    # run the sim and save the bunch
    if run:
        pre = 'temp_beam/'
        suf = 'temp'
        if locationsToSave == []:
            locationsToSave = [start, finish]
        if edited_bunch_energy_at_checkpoints_MeV!=[None, None, None, None]:
            tao = run_initialized_sim_edit_bunch_energy(tao, start, finish, edited_bunch_energy_at_checkpoints_MeV=edited_bunch_energy_at_checkpoints_MeV, pre=pre, suf=suf, locations=locationsToSave)
        else:
            tao = run_initialized_sim(tao, locationsToSave[0], locationsToSave[-1], pre, suf, locationsToSave, desired_P0Cs_MeV=desired_P0Cs_MeV)
    return tao

def set_beam(tao, file, numMacroParticles = None, timeCenterTF=True):
    '''
    Sets the beam in the tao object. The beam is edited: drift to z, z=0. t=0 if time_centering.
    '''
    #filePath = os.getcwd()
    #file_e = f'{filePath}/beams/activeBeamFile.h5'
    file_e = file + "_e"
    #Write as the active file
    #modifyInputBeamSimple(folder + file + ".h5", numMacroParticles).write(file_e + ".h5")
    modifyInputBeamSimple(file + ".h5", numMacroParticles, timeCenterTF=timeCenterTF).write(file_e + ".h5")
    #tao.cmd(f'set beam_init position_file={folder + file_e + ".h5"}')
    tao.cmd(f'set beam_init position_file={file_e + ".h5"}')
    tao.cmd('reinit beam')

def run_initialized_sim(tao, start, finish, pre='temp_beam/', suf='temp', locations=[], treat_dipoles=False, desired_P0Cs_MeV=[None,None,None,None]):
    '''
    If desired_P0Cs_MeV are None and treat_dipoles is False, this function just tracks from start to finish, saving the beam at the locations in "locations".
    If treat_dipoles is true, before tracking it loads the constant fields from a "nominal" experiment lattice.
    If desired_P0Cs_MeV is true:
    tunes the cavities to the provided edited_bunch_energy_at_checkpoints_MeV
    tracks to the BX0FBEG
    tunes them back to 125, 335, 4500, 10000
    tracks to the BX0FEND
    tunes them to the provided edited_bunch_energy_at_checkpoints_MeV
    tracks to the BC11CBEG
    tunes them back to 125, 335, 4500, 10000
    tracks to the BC11CEND
    tunes them to the provided edited_bunch_energy_at_checkpoints_MeV
    tracks to the BC14CBEG
    tunes them back to 125, 335, 4500, 10000
    tracks to the BC14CEND
    tunes them to the provided edited_bunch_energy_at_checkpoints_MeV
    tracks to the BC20CBEG
    tunes them back to 125, 335, 4500, 10000
    tracks to the BC20CEND
    tunes them to the provided edited_bunch_energy_at_checkpoints_MeV
    tracks to the finish

    '''
    if locations==[]:
        locations = [start, finish]
    if locations==None:
        locations=[]

    if treat_dipoles:
        tao = treat_dipoles1(tao)
        trackBeam(tao, filepath=tao.filePathGlobal, trackStart = start, trackEnd = finish, autoLoadActiveFile=False)
    else:
        current_start = start
        # injector
        if desired_P0Cs_MeV[0] is not None:
            tune_to_P0Cs(tao, desired_P0Cs_MeV=desired_P0Cs_MeV, change_only_L0B=True)
            trackBeam(tao, filepath=tao.filePathGlobal, trackStart = current_start, trackEnd = "BX0FBEG", autoLoadActiveFile=False)
            getBeamAtElement(tao, "BX0FBEG", tToZ=False).write(tao.filePathGlobal+"/"+"temp_beam/temp.h5")
            set_beam(tao, tao.filePathGlobal+"/"+"temp_beam/temp")
            tune_to_P0Cs(tao, desired_P0Cs_MeV=[125, 335, 4500, 10000], change_only_L0B=True)        
            current_start = "BX0FBEG"
            trackBeam(tao, filepath=tao.filePathGlobal, trackStart = current_start, trackEnd = "BX0FEND", autoLoadActiveFile=False)
            #print(f'<x> inside of the dogleg: {tao.bunch_params("BPM10731")["centroid_vec_1"]}')
            getBeamAtElement(tao, "BX0FEND", tToZ=False).write(tao.filePathGlobal+"/"+"temp_beam/temp.h5")
            set_beam(tao, tao.filePathGlobal+"/"+"temp_beam/temp")
            current_start = "BX0FEND"
            tune_to_P0Cs(tao, desired_P0Cs_MeV=desired_P0Cs_MeV)
        # L1
        if desired_P0Cs_MeV[1] is not None:
            trackBeam(tao, filepath=tao.filePathGlobal, trackStart = current_start, trackEnd = "BC11CBEG", autoLoadActiveFile=False)
            getBeamAtElement(tao, "BC11CBEG", tToZ=False).write(tao.filePathGlobal+"/"+"temp_beam/temp.h5")
            set_beam(tao, tao.filePathGlobal+"/"+"temp_beam/temp")
            tune_to_P0Cs(tao, desired_P0Cs_MeV=[125, 335, 4500, 10000])
            current_start = "BC11CBEG"
            trackBeam(tao, filepath=tao.filePathGlobal, trackStart = current_start, trackEnd = "BC11CEND", autoLoadActiveFile=False)
            getBeamAtElement(tao, "BC11CEND", tToZ=False).write(tao.filePathGlobal+"/"+"temp_beam/temp.h5")
            set_beam(tao, tao.filePathGlobal+"/"+"temp_beam/temp")
            current_start = "BC11CEND"
            tune_to_P0Cs(tao, desired_P0Cs_MeV=desired_P0Cs_MeV)
        # L2
        if desired_P0Cs_MeV[2] is not None:
            trackBeam(tao, filepath=tao.filePathGlobal, trackStart = current_start, trackEnd = "BEGBC14E", autoLoadActiveFile=False)
            getBeamAtElement(tao, "BEGBC14E", tToZ=False).write(tao.filePathGlobal+"/"+"temp_beam/temp.h5")
            set_beam(tao, tao.filePathGlobal+"/"+"temp_beam/temp")
            tune_to_P0Cs(tao, desired_P0Cs_MeV=[125, 335, 4500, 10000])
            current_start = "BEGBC14E"
            trackBeam(tao, filepath=tao.filePathGlobal, trackStart = current_start, trackEnd = "ENDBC14E", autoLoadActiveFile=False)
            getBeamAtElement(tao, "ENDBC14E", tToZ=False).write(tao.filePathGlobal+"/"+"temp_beam/temp.h5")
            set_beam(tao, tao.filePathGlobal+"/"+"temp_beam/temp")
            current_start = "ENDBC14E"
            tune_to_P0Cs(tao, desired_P0Cs_MeV=desired_P0Cs_MeV)
        # L3
        if desired_P0Cs_MeV[3] is not None:
            trackBeam(tao, filepath=tao.filePathGlobal, trackStart = current_start, trackEnd = "BEGBC20", autoLoadActiveFile=False)
            getBeamAtElement(tao, "BEGBC20", tToZ=False).write(tao.filePathGlobal+"/"+"temp_beam/temp.h5")
            set_beam(tao, tao.filePathGlobal+"/"+"temp_beam/temp")
            tune_to_P0Cs(tao, desired_P0Cs_MeV=[125, 335, 4500, 10000])
            current_start = "BEGBC20"
            trackBeam(tao, filepath=tao.filePathGlobal, trackStart = current_start, trackEnd = "ENDBC20", autoLoadActiveFile=False)
            getBeamAtElement(tao, "ENDBC20", tToZ=False).write(tao.filePathGlobal+"/"+"temp_beam/temp.h5")
            set_beam(tao, tao.filePathGlobal+"/"+"temp_beam/temp")
            current_start = "ENDBC20"
            tune_to_P0Cs(tao, desired_P0Cs_MeV=desired_P0Cs_MeV)
        # Fin
        trackBeam(tao, filepath=tao.filePathGlobal, trackStart = current_start, trackEnd = finish, autoLoadActiveFile=False)

    for ind in range(len(locations)):
        P = getBeamAtElement(tao, locations[ind], tToZ=False)
        P.write(tao.filePathGlobal+"/"+pre+locations[ind]+suf +'.h5')

    return tao


def run_initialized_sim_edit_bunch_energy(tao, start, finish, pre='temp_beam/', suf='temp', locations=[], edited_bunch_energy_at_checkpoints_MeV=[None,None,None,None]):
    '''
    Tracks the bunch and changes the bunch energy at BX0FBEG, BC11CBEG, ENDL2F, ENDL3F_2 if any of edited_bunch_energy_at_checkpoints_MeV (list with 4 numbers) is not -1.
    '''
    if locations==[]:
        locations = [start, finish]
    if locations==None:
        locations=[]
    current_start = start
    if edited_bunch_energy_at_checkpoints_MeV[0] is not None:
        trackBeam(tao, filepath=tao.filePathGlobal, trackStart = current_start, trackEnd = "BX0FBEG", autoLoadActiveFile=False)
        edit_bunch_parameters_from_PG(getBeamAtElement(tao, "BX0FBEG", tToZ=False), pzMeV=edited_bunch_energy_at_checkpoints_MeV[0], means=[None, None, None, None, None], path_to_write=tao.filePathGlobal+"/"+"temp_beam/temp")
        current_start = "BX0FBEG"
        set_beam(tao, tao.filePathGlobal+"/"+"temp_beam/temp")

    if edited_bunch_energy_at_checkpoints_MeV[1] is not None:
        trackBeam(tao, filepath=tao.filePathGlobal, trackStart = current_start, trackEnd = "BC11CBEG", autoLoadActiveFile=False)
        edit_bunch_parameters_from_PG(getBeamAtElement(tao, "BC11CBEG", tToZ=False), pzMeV=edited_bunch_energy_at_checkpoints_MeV[1], means=[None, None, None, None, None], path_to_write=tao.filePathGlobal+"/"+"temp_beam/temp")
        current_start = "BC11CBEG"
        set_beam(tao, tao.filePathGlobal+"/"+"temp_beam/temp")

    if edited_bunch_energy_at_checkpoints_MeV[2] is not None:
        trackBeam(tao, filepath=tao.filePathGlobal, trackStart = current_start, trackEnd = "ENDL2F", autoLoadActiveFile=False)
        edit_bunch_parameters_from_PG(getBeamAtElement(tao, "ENDL2F", tToZ=False), pzMeV=edited_bunch_energy_at_checkpoints_MeV[2], means=[None, None, None, None, None], path_to_write=tao.filePathGlobal+"/"+"temp_beam/temp")
        current_start = "ENDL2F"
        set_beam(tao, tao.filePathGlobal+"/"+"temp_beam/temp")

    if edited_bunch_energy_at_checkpoints_MeV[3] is not None:
        trackBeam(tao, filepath=tao.filePathGlobal, trackStart = current_start, trackEnd = "ENDL3F_2", autoLoadActiveFile=False)
        edit_bunch_parameters_from_PG(getBeamAtElement(tao, "ENDL3F_2", tToZ=False), pzMeV=edited_bunch_energy_at_checkpoints_MeV[3], means=[None, None, None, None, None], path_to_write=tao.filePathGlobal+"/"+"temp_beam/temp")
        current_start = "ENDL3F_2"
        set_beam(tao, tao.filePathGlobal+"/"+"temp_beam/temp")

    
    trackBeam(tao, filepath=tao.filePathGlobal, trackStart = current_start, trackEnd = finish, autoLoadActiveFile=False)

    for ind in range(len(locations)):
        P = getBeamAtElement(tao, locations[ind], tToZ=False)
        P.write(tao.filePathGlobal+"/"+pre+locations[ind]+suf +'.h5')

    return tao


### Correct the lattice to match the desired Pz

def tune_to_P0Cs(tao, desired_P0Cs_MeV=[125, 335, 4500, 10000], change_only_L0B=False):
    '''
    This function scales the cavities to match the desired_P0Cs_MeV.
    change_only_L0B: see the "edit_energy_based_on_beam_inj" function.
    '''
    desired_P0Cs_MeV = [desired_P0Cs_MeV[0] if desired_P0Cs_MeV[0] is not None else 125,
                        desired_P0Cs_MeV[1] if desired_P0Cs_MeV[1] is not None else 335,
                        desired_P0Cs_MeV[2] if desired_P0Cs_MeV[2] is not None else 4500,
                        desired_P0Cs_MeV[3] if desired_P0Cs_MeV[3] is not None else 10000]
    if change_only_L0B:
        current_pz_location = tao.ele_gen_attribs('BX0FBEG')["P0C"]*1e-6
        current_pz_start = tao.ele_gen_attribs('L0AFEND')["P0C"]*1e-6
        coef_e = 1 + (desired_P0Cs_MeV[0] - current_pz_location)/(current_pz_location-current_pz_start)
        l0bVoltage = tao.ele_gen_attribs('L0BF')["VOLTAGE"]
        tao.cmd(f'set ele L0BF VOLTAGE = {l0bVoltage*coef_e}')
    else:
        current_pz_location = tao.ele_gen_attribs('BX0FBEG')["P0C"]*1e-6
        current_pz_start = tao.ele_gen_attribs('BEGINNING')["P0C"]*1e-6
        coef_e = 1 + (desired_P0Cs_MeV[0] - current_pz_location)/(current_pz_location-current_pz_start)
        l0aVoltage = tao.ele_gen_attribs('L0AF')["VOLTAGE"]
        l0bVoltage = tao.ele_gen_attribs('L0BF')["VOLTAGE"]
        tao.cmd(f'set ele L0AF VOLTAGE = {l0aVoltage*coef_e}')
        tao.cmd(f'set ele L0BF VOLTAGE = {l0bVoltage*coef_e}')

    setLinacGradientAuto(tao, "L1", (desired_P0Cs_MeV[1]-desired_P0Cs_MeV[0])*1e6)
    setLinacGradientAuto(tao, "L2", (desired_P0Cs_MeV[2]-desired_P0Cs_MeV[1])*1e6)
    setLinacGradientAuto(tao, "L3", (desired_P0Cs_MeV[3]-desired_P0Cs_MeV[2])*1e6)

    return tao


#### Treat dipoles

def save_dipoles(tao, dipoleEnergiesMeV=[125, 335, 4500, 10000]):
    '''
    A test function to deal with the dipoles. Hopefully I will find the solution...
    '''
    initial_energy_bx = tao.ele_gen_attribs('BX0FBEG')["P0C"]*1e-6
    initial_energy_bc11 = tao.ele_gen_attribs('BC11CBEG')["P0C"]*1e-6
    initial_energy_bc14 = tao.ele_gen_attribs('ENDL2F')["P0C"]*1e-6
    initial_energy_bc20 = tao.ele_gen_attribs('ENDL3F_2')["P0C"]*1e-6

    tao = tune_to_P0Cs(tao, desired_P0Cs_MeV=dipoleEnergiesMeV)

    fields = {}

    elems = get_element_array(tao, "BEGINNING", "END", values_to_show=["SBend"])[:,1]
    for ele in elems:
        fields[ele] = tao.ele_gen_attribs(ele)["B_FIELD"]

    # Change the energies back
    tao = tune_to_P0Cs(tao, desired_P0Cs_MeV=[initial_energy_bx,initial_energy_bc11,initial_energy_bc14,initial_energy_bc20])

    return fields

def treat_dipoles(tao, fields):
    '''
    A function to deal with the dipoles. Uses the fields from the input.
    Use this function after all other adjustments (before running the simulation) to set the fields to the nominal values.
    WARNING! Will not work properly if the dipoles were set to field_master=True at any point.
    '''
    for k, v in fields.items():
        current_field = tao.ele_gen_attribs(k)["B_FIELD"]
        tao.cmd(f'set ele {k} DB_FIELD = {v-current_field}')
    return tao

fields = {
    'BCX10451': 0.4399498913496881,
    'BCX10461': -0.4399498913496881,
    'BCX10475': -0.4399498913496881,
    'BCX10481': 0.4399498913496881,
    'BX10661': 0.6242944753331842,
    'BX10751': 0.6242944753331842,
    'BCX11314': 0.5167328829297726,
    'BCX11331': -0.5167328829297726,
    'BCX11338': -0.5167328829297726,
    'BCX11355': 0.5167328829297726,
    'BCX14720': 1.145800533192734,
    'BCX14796': -1.145800533192734,
    'BCX14808': -1.145800533192734,
    'BCX14883': 1.145800533192734,
    'B1LE': -0.7088897708589302,
    'B2LE': 0.5998061022132172,
    'B3LE': -0.6450166589406902,
    'B3RE': -0.6450166589406902,
    'B2RE': 0.5998061022132172,
    'WIGE1': 0.3417645361988085,
    'WIGE2': -0.3417645361988085,
    'WIGE3': 0.3417645361988085,
    'B1RE': -0.7088897708589302,
    'B5D36': -0.2046605187706695
}

def treat_dipoles1(tao):
    '''
    A function to deal with the dipoles. Uses the constants for the fields tuned to the nominal energies.
    Use this function after all other adjustments (before running the simulation) to set the fields to the nominal values.
    WARNING! Will not work properly if the dipoles were set to field_master=True at any point.
    '''
    for k, v in fields.items():
        current_field = tao.ele_gen_attribs(k)["B_FIELD"]
        tao.cmd(f'set ele {k} DB_FIELD = {v-current_field}')
    return tao


#### Adjust based on the beam


def edit_energy_based_on_beam_inj(tao, location="BX0FBEG", desiredPzMeV=125, change_only_L0B=False):
    '''
    Scales the injector cavities to match the desired beam energy at the dogleg (desiredPzMeV) in MeV.
    location: where to look at the beam energy (it changes with s if wake fields or CSR are enabled)
    change_only_L0B: scale only L0B. I think that at FACET we do exactly that.
    If changing both L0A and L0B, the input beam energy at L0A is adjusted accordingly later.
    '''
    if change_only_L0B:
        current_pz_location = np.mean(getBeamAtElement(tao, location, tToZ=False).pz)
        current_pz_start = np.mean(getBeamAtElement(tao, "L0AFEND", tToZ=False).pz)
        coef_e = 1 + 1e6*(desiredPzMeV - current_pz_location*1e-6)/(current_pz_location-current_pz_start)
        l0bVoltage = tao.ele_gen_attribs('L0BF')["VOLTAGE"]
        tao.cmd(f'set ele L0BF VOLTAGE = {l0bVoltage*coef_e}')
    else:
        current_pz_location = np.mean(getBeamAtElement(tao, location, tToZ=False).pz)
        current_pz_start = tao.ele_gen_attribs('BEGINNING')["P0C"]
        #current_pz_location = tao.ele_gen_attribs(location)["P0C"]
        coef_e = 1 + 1e6*(desiredPzMeV - current_pz_location*1e-6)/(current_pz_location-current_pz_start)
        l0aVoltage = tao.ele_gen_attribs('L0AF')["VOLTAGE"]
        l0bVoltage = tao.ele_gen_attribs('L0BF')["VOLTAGE"]
        tao.cmd(f'set ele L0AF VOLTAGE = {l0aVoltage*coef_e}')
        tao.cmd(f'set ele L0BF VOLTAGE = {l0bVoltage*coef_e}')
    return tao

def edit_energy_based_on_beam_L1(tao, location="BC11CBEG", desiredPzMeV=335):
    '''
    Scales the L1 cavities to match the desired beam energy at the dogleg (desiredPzMeV) in MeV.
    location: where to look at the beam energy (it changes with s if wake fields or CSR are enabled)
    '''
    current_pz_location = np.mean(getBeamAtElement(tao, location, tToZ=False).pz)
    current_pz_start = np.mean(getBeamAtElement(tao, 'BX0FBEG', tToZ=False).pz)
    activeMatchStrings = matchStringWrapper(tao, "L1")
    coef_e = 1 + 1e6*(desiredPzMeV - current_pz_location*1e-6)/(current_pz_location-current_pz_start)
    for i in activeMatchStrings:
        g = tao.ele_gen_attribs(i)["GRADIENT"]
        tao.cmd(f'set ele {i} GRADIENT = {g*coef_e}')
    return tao

def edit_energy_based_on_beam_L2(tao, location="ENDL2F", desiredPzMeV=4500):
    '''
    Scales the L2 cavities to match the desired beam energy at the dogleg (desiredPzMeV) in MeV.
    location: where to look at the beam energy (it changes with s if wake fields or CSR are enabled)
    '''
    current_pz_location = np.mean(getBeamAtElement(tao, location, tToZ=False).pz)
    current_pz_start = np.mean(getBeamAtElement(tao, 'BC11CBEG', tToZ=False).pz)
    activeMatchStrings = matchStringWrapper(tao, "L2")
    coef_e = 1 + 1e6*(desiredPzMeV - current_pz_location*1e-6)/(current_pz_location-current_pz_start)
    for i in activeMatchStrings:
        g = tao.ele_gen_attribs(i)["GRADIENT"]
        tao.cmd(f'set ele {i} GRADIENT = {g*coef_e}')
    return tao

def edit_energy_based_on_beam_L3(tao, location="ENDL3F_2", desiredPzMeV=10000):
    '''
    Scales the L3 cavities to match the desired beam energy at the dogleg (desiredPzMeV) in MeV.
    location: where to look at the beam energy (it changes with s if wake fields or CSR are enabled)
    '''
    current_pz_location = np.mean(getBeamAtElement(tao, location, tToZ=False).pz)
    current_pz_start = np.mean(getBeamAtElement(tao, 'ENDL2F', tToZ=False).pz)
    activeMatchStrings = matchStringWrapper(tao, "L3")
    coef_e = 1 + 1e6*(desiredPzMeV - current_pz_location*1e-6)/(current_pz_location-current_pz_start)
    for i in activeMatchStrings:
        g = tao.ele_gen_attribs(i)["GRADIENT"]
        tao.cmd(f'set ele {i} GRADIENT = {g*coef_e}')
    return tao

def edit_energy_based_on_beam_all(tao, start, file, desired_beam_energies=None, change_only_L0B=False, change_file_pz=True, verbose=False, finalnumMacroParticles=5e4):
    '''
    This function changes the cavity voltages so that the tracked beam has the "desired_P0Cs" <pz> between the cavities.
    desired_P0Cs: the desired <pz> in eV between the cavities. Must be None or a list of 4 numbers (dogleg, bc11, bc14, bc20).
    If None, the P0Cs from the lattice are used.
    start: start of the simulation. Need to be in the injector (use L0AFEND to avoid confusion).
    file: the bunch file to use for the tuning. Providing it is a must because the energy obviously depends on the charge and the size of the bunch.
    change_only_L0B: in the injector, scale only L0B. I think that at FACET we do exactly that.
    If changing both L0A and L0B, the input beam energy at L0A is adjusted accordingly later.
    change_file_pz: change the bunch energy at the "start" to match the "start" P0C.
    finalnumMacroParticles: the function uses "set_beam()" at the end. If you don't want to set the beam manually after the tuning,
    this option allows you to regulate the number of particles in that set beam (the one you provided with the "file").
    '''
    if desired_beam_energies is None:
        desired_beam_energies = [float(tao.ele_gen_attribs("BX0FBEG")["P0C"]), float(tao.ele_gen_attribs("BC11CBEG")["P0C"]), float(tao.ele_gen_attribs("ENDL2F")["P0C"]), float(tao.ele_gen_attribs("ENDL3F_2")["P0C"])]
    pre = 'temp_beam/'
    suf = 'temp'
    if verbose:
        print(f'P0C now: {desired_beam_energies}')
    
    # injector
    if change_file_pz:
        edit_bunch_parameters(file, pzMeV=tao.ele_gen_attribs(start)["P0C"]*1e-6, moments=[None,None,None,None,None,None], means=[0,0,0,0,0], charge=-1, path_to_write=file)
    set_beam(tao, file, numMacroParticles = 5e4)
    locations = [start, "BX0FBEG"]
    tao = run_initialized_sim(tao, locations[0], locations[-1], pre, suf, locations)
    if verbose:
        print(f'beam to BX0FBEG: {[float(np.mean(getBeamAtElement(tao, "BX0FBEG", tToZ=False).pz))]}')
    
    tao = edit_energy_based_on_beam_inj(tao, location=locations[-1], desiredPzMeV=desired_beam_energies[0]*1e-6, change_only_L0B=change_only_L0B)
    if verbose:
        print(f'edited P0C BX0FBEG: {tao.ele_gen_attribs("BX0FBEG")["P0C"]}')
    
    # L1
    if change_file_pz:
        edit_bunch_parameters(file, pzMeV=tao.ele_gen_attribs(start)["P0C"]*1e-6, moments=[None,None,None,None,None,None], means=[0,0,0,0,0], charge=-1, path_to_write=file)
    set_beam(tao, file, numMacroParticles = 5e4)
    locations = [start, "BC11CBEG"]
    tao = run_initialized_sim(tao, locations[0], locations[-1], pre, suf, locations)
    if verbose:
        print(f'beam to BC11CBEG: {[float(np.mean(getBeamAtElement(tao, "BX0FBEG", tToZ=False).pz)), float(np.mean(getBeamAtElement(tao, "BC11CBEG", tToZ=False).pz))]}')
    
    tao = edit_energy_based_on_beam_L1(tao, location=locations[-1], desiredPzMeV=desired_beam_energies[1]*1e-6)
    if verbose:
        print(f'edited P0C BC11CBEG: {tao.ele_gen_attribs("BC11CBEG")["P0C"]}')
    
    # L2
    if change_file_pz:
        edit_bunch_parameters(file, pzMeV=tao.ele_gen_attribs(start)["P0C"]*1e-6, moments=[None,None,None,None,None,None], means=[0,0,0,0,0], charge=-1, path_to_write=file)
    set_beam(tao, file, numMacroParticles = 5e4)
    locations = [start, "ENDL2F"]
    tao = run_initialized_sim(tao, locations[0], locations[-1], pre, suf, locations)
    if verbose:
        print(f'beam to ENDL2F: {[float(np.mean(getBeamAtElement(tao, "BX0FBEG", tToZ=False).pz)), float(np.mean(getBeamAtElement(tao, "BC11CBEG", tToZ=False).pz)), float(np.mean(getBeamAtElement(tao, "ENDL2F", tToZ=False).pz))]}')
    
    tao = edit_energy_based_on_beam_L2(tao, location=locations[-1], desiredPzMeV=desired_beam_energies[2]*1e-6)
    if verbose:
        print(f'edited P0C ENDL2F: {tao.ele_gen_attribs("ENDL2F")["P0C"]}')
    
    # L3
    if change_file_pz:
        edit_bunch_parameters(file, pzMeV=tao.ele_gen_attribs(start)["P0C"]*1e-6, moments=[None,None,None,None,None,None], means=[0,0,0,0,0], charge=-1, path_to_write=file)
    set_beam(tao, file, numMacroParticles = 5e4)
    locations = [start, "ENDL3F_2"]
    tao = run_initialized_sim(tao, locations[0], locations[-1], pre, suf, locations)
    if verbose:
        print(f'beam to ENDL3F_2: {[float(np.mean(getBeamAtElement(tao, "BX0FBEG", tToZ=False).pz)), float(np.mean(getBeamAtElement(tao, "BC11CBEG", tToZ=False).pz)), float(np.mean(getBeamAtElement(tao, "ENDL2F", tToZ=False).pz)), float(np.mean(getBeamAtElement(tao, "ENDL3F_2", tToZ=False).pz))]}')
    
    tao = edit_energy_based_on_beam_L3(tao, location=locations[-1], desiredPzMeV=desired_beam_energies[3]*1e-6)
    if verbose:
        print(f'edited P0C ENDL3F_2: {tao.ele_gen_attribs("ENDL3F_2")["P0C"]}')
    
    set_beam(tao, file, numMacroParticles=finalnumMacroParticles)
    return tao


## Scans

def make_1d_scan(tao, mean=0, nscan=21, scan_span=5e-3, function_to_change_tao_in_scan=None, function_to_get_results_in_scan=None, plot=True, label="y", xlabel="x axis", ylabel="y axis", **kwargs):
    """Perform a 1D parameter scan over Tao and optionally plot the results.
    """
    scan_values = scan_span*(np.arange(nscan)-(nscan-1)/2)/(nscan-1)+mean
    output = []
    for value_ind in range(nscan):
        tao = function_to_change_tao_in_scan(tao, scan_values[value_ind], **kwargs)
        output.append(function_to_get_results_in_scan(tao, **kwargs))
    output = np.array(output)
    if plot:
        make_a_plot(scan_values, output, label=label, x_label=xlabel, y_label=ylabel, cartesian_axes=[False, False], axes_location=[0, 0])
    return scan_values, output


def make_comparison_dz_2nd_order(tao, beam_file='temp_beam/temp', start='L0AFEND', finish='PR11375', means_shift=[0,0,0,0,0,0], theory=True, second_order=True, **kwargs):
    """Compare simulation and second-order theory for longitudinal bunch size growth."""
    # sim
    set_beam(tao, beam_file)
    run_initialized_sim(tao, start, finish)
    
    dzsim = 3e8*(getBeamAtElement(tao, finish, tToZ=False).t - np.mean(getBeamAtElement(tao, finish, tToZ=False).t))
    size_sim = np.sqrt(np.mean(dzsim**2) - np.mean(dzsim)**2)

    # theory
    if theory:
        beam = make_simple_bunch_theory_from_bunch_sims(beam_file+"_e", np.mean(ParticleGroup(beam_file+"_e.h5").pz)*1e-6, means_shift=means_shift)
        
        r5i = np.array([float(get_rij(tao, start, finish, 5, j+1)) for j in range(6)])
        t5ij = np.zeros((6,6))
        for j in range(6):
            for k in range(6):
                t5ij[j][k] = float(get_tijk(tao, start, finish, 5, j+1, k+1))
        nonlinear_impact = np.zeros(len(beam))
        for j in range(6):
            for k in range(6):
                nonlinear_impact += (t5ij[j][k]*(beam[:, j]*beam[:, k]) if j>=k else 0)/2
        nonlinear_impact = nonlinear_impact if second_order else 0*nonlinear_impact
        dz = np.sum([r5i[j]*beam[:, j] for j in range(6)], axis=0) + nonlinear_impact
        dz = dz - np.mean(dz)
        size_theory = np.sqrt(np.mean(dz**2) - np.mean(dz)**2)
        
        return size_sim, size_theory
    else:
        return size_sim
    

## Simulation parameters functions

def get_element_array(tao, beg, end, values_to_show=[], values_to_remove=[], marginl=0, marginr=0):
    """Return a subset of lattice elements between two locations, filtered by element type."""
    keys = tao.lat_list("*", "ele.key")
    ss = tao.lat_list("*", "ele.s")
    names = tao.lat_list("*", "ele.name")
    
    elements = np.stack([ss, names, keys], axis=1)
    
    location1 = np.argwhere(elements==beg)[0,0]-marginl
    location2 = np.argwhere(elements==end)[0,0]+marginr
    
    trunc_array = elements[location1:location2+1]
    show_array = trunc_array[np.isin(trunc_array[:, 2], values_to_show)] if (len(values_to_show)>0) else trunc_array
    clean_array = show_array[~np.isin(trunc_array[:, 2], values_to_remove)] if (len(values_to_remove)>0) else show_array
    
    return clean_array

def get_tijk(tao, loc1, loc2, i_0, j_0, k_0):
    """Read a second-order Taylor map coefficient (T) from Tao between two locations."""
    n = 6
    t5ijterms = [
        [
            tuple(1 if k == i or k == j else 0 for k in range(n)) if i != j
            else tuple(2 if k == i else 0 for k in range(n))
            for j in range(n)
        ]
        for i in range(n)
    ]
    mapterms = tao.taylor_map(loc1, loc2, order='2', verbose=False, as_dict=True, raises=True)[i_0]
    return mapterms[t5ijterms[j_0-1][k_0-1]] if t5ijterms[j_0-1][k_0-1] in mapterms else 0

def get_rij(tao, loc1, loc2, i, j):
    """Read an R-matrix element from Tao between two locations."""
    r5i = np.zeros(6)
    s = tao.cmd("".join(["show matrix ", loc1, " ", loc2]))[i+1]
    numeric_part = s.split(':')[0]
    nums = [float(x) for x in numeric_part.split()]
    r5i = np.array(nums)
    return float(r5i[j-1])


### Sextupole settings

def setAllWChicaneSextupoles(tao, S1ELkG, S2ELkG, S3ELkG, S3ERkG, S2ERkG, S1ERkG):
    """Set all chicane sextupole strengths in the Tao lattice."""
    setSextkG(tao, "S1EL",   S1ELkG)
    setSextkG(tao, "S2EL",   S2ELkG)
    setSextkG(tao, "S3EL_1", S3ELkG)
    setSextkG(tao, "S3EL_2", S3ELkG)
    setSextkG(tao, "S3ER_1", S3ERkG)
    setSextkG(tao, "S3ER_2", S3ERkG)
    setSextkG(tao, "S2ER",   S2ERkG)
    setSextkG(tao, "S1ER",   S1ERkG)
    return tao

def setAllWChicaneSextupolesXOffsets(tao, S1EL_dx, S2EL_dx, S3EL_dx, S3ER_dx, S2ER_dx, S1ER_dx):
    """Set all chicane sextupole horizontal offsets in the Tao lattice."""
    tao.cmd(f'set ele {"S1EL"} X_OFFSET = {S1EL_dx}')
    tao.cmd(f'set ele {"S2EL"} X_OFFSET = {S2EL_dx}')
    tao.cmd(f'set ele {"S3EL_1"} X_OFFSET = {S3EL_dx}')
    tao.cmd(f'set ele {"S3EL_2"} X_OFFSET = {S3EL_dx}')
    tao.cmd(f'set ele {"S3ER_1"} X_OFFSET = {S3ER_dx}')
    tao.cmd(f'set ele {"S3ER_2"} X_OFFSET = {S3ER_dx}')
    tao.cmd(f'set ele {"S2ER"} X_OFFSET = {S2ER_dx}')
    tao.cmd(f'set ele {"S1ER"} X_OFFSET = {S1ER_dx}')
    return tao

def setAllWChicaneSextupolesYOffsets(tao, S1EL_dx, S2EL_dx, S3EL_dx, S3ER_dx, S2ER_dx, S1ER_dx):
    """Set all chicane sextupole vertical offsets in the Tao lattice."""
    tao.cmd(f'set ele {"S1EL"} Y_OFFSET = {S1EL_dx}')
    tao.cmd(f'set ele {"S2EL"} Y_OFFSET = {S2EL_dx}')
    tao.cmd(f'set ele {"S3EL_1"} Y_OFFSET = {S3EL_dx}')
    tao.cmd(f'set ele {"S3EL_2"} Y_OFFSET = {S3EL_dx}')
    tao.cmd(f'set ele {"S3ER_1"} Y_OFFSET = {S3ER_dx}')
    tao.cmd(f'set ele {"S3ER_2"} Y_OFFSET = {S3ER_dx}')
    tao.cmd(f'set ele {"S2ER"} Y_OFFSET = {S2ER_dx}')
    tao.cmd(f'set ele {"S1ER"} Y_OFFSET = {S1ER_dx}')
    return tao
