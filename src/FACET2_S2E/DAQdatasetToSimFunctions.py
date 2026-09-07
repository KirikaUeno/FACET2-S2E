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

"""Functions to edit a Tao lattice according to a FACET-II DAQ experiment dataset.

This module holds the BMAD-element-to-EPICS-PV maps (quadrupoles,
sextupoles, cavities, correctors, bends) and the functions that read a
DAQ database (DATASET) and push those values into the lattice, either
just for the energy profile or for the full magnet/corrector settings.

Split out of the former functionsForSims.py.
"""

## Edit lattice according to experiment

### BMAD to DAQ maps

#quadrupoles
bmad_quad_to_pv_map = {
    # s10
    'QA10361': ["nonBSA_List_S10", 'QUAD_IN10_361_BDES'],
    'QA10371': ["nonBSA_List_S10", 'QUAD_IN10_371_BDES'],
    'QE10425': ["nonBSA_List_S10", 'QUAD_IN10_425_BDES'],
    'QE10441': ["nonBSA_List_S10", 'QUAD_IN10_441_BDES'],
    'QE10511': ["nonBSA_List_S10", 'QUAD_IN10_511_BDES'],
    'QE10525': ["nonBSA_List_S10", 'QUAD_IN10_525_BDES'],
    'QM10631': ["nonBSA_List_S10", 'QUAD_IN10_631_BDES'],
    'QM10651': ["nonBSA_List_S10", 'QUAD_IN10_651_BDES'],
    'QB10731': ["nonBSA_List_S10", 'QUAD_IN10_731_BDES'],
    'QM10771': ["nonBSA_List_S10", 'QUAD_IN10_771_BDES'],
    'QM10781': ["nonBSA_List_S10", 'QUAD_IN10_781_BDES'],
    
    # s11
    'QA11132': ["nonBSA_List_S11", 'QUAD_LI11_132_BCON'],
    'Q11201': ["nonBSA_List_S11", 'QUAD_LI11_201_BCON'],
    'QA11265': ["nonBSA_List_S11", 'QUAD_LI11_265_BCON'],
    'Q11301': ["nonBSA_List_S11", 'QUAD_LI11_301_BCON'],
    'QM11312': ["nonBSA_List_S11", 'QUAD_LI11_312_BCON'],
    'CQ11317': ["nonBSA_List_S11", 'QUAD_LI11_317_BCON'],
    'SQ11340': ["nonBSA_List_S11", 'QUAD_LI11_340_BCON'],
    'CQ11352': ["nonBSA_List_S11", 'QUAD_LI11_352_BCON'],
    'QM11358': ["nonBSA_List_S11", 'QUAD_LI11_358_BCON'],
    'QM11362': ["nonBSA_List_S11", 'QUAD_LI11_362_BCON'],
    'QM11393': ["nonBSA_List_S11", 'QUAD_LI11_393_BCON'],

    # s12 - s18
    # NOT SAVED IN DAQ?

    # s19
    # MOST ARE NOT SAVED IN DAQ?
    'Q19851': ["nonBSA_List_S19", 'QUAD_LI19_851_BACT'],
    'Q19871': ["nonBSA_List_S19", 'QUAD_LI19_871_BACT'],

    # s20
    'SQ1': ['nonBSA_List_S20Magnets', 'LI20_QUAD_2086_BACT'],
    'Q1EL': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2060_BACT'],
    'Q2EL': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2130_BACT'],
    'Q3EL_1': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2150_BACT'],
    'Q3EL_2': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2150_BACT'],
    'Q4EL_1': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2200_BACT'],
    'Q4EL_2': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2200_BACT'],
    'Q4EL_3': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2200_BACT'],
    'Q5EL': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2230_BACT'],
    'Q6E': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2251_BACT'],
    'Q5ER': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2230_BACT'],
    'Q4ER_1': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2200_BACT'],
    'Q4ER_2': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2200_BACT'],
    'Q4ER_3': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2200_BACT'],
    'Q3ER_1': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2150_BACT'],
    'Q3ER_2': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2150_BACT'],
    'Q2ER': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2130_BACT'],
    'Q1ER': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2060_BACT'],
    #'SQ2': ['nonBSA_List_S20Magnets', 'LI20_QUAD_3015_BACT'], SQ2 is not in the BMAD model. But it is usually 0 anyway.
    'Q5FF': ['nonBSA_List_S20Magnets', 'LI20_LGPS_3011_BACT'],
    'Q4FF': ['nonBSA_List_S20Magnets', 'LI20_LGPS_3311_BACT'],
    'Q3FF': ['nonBSA_List_S20Magnets', 'LI20_LGPS_3151_BACT'],
    'Q2FF': ['nonBSA_List_S20Magnets', 'LI20_LGPS_1910_BACT'],
    'Q1FF': ['nonBSA_List_S20Magnets', 'LI20_LGPS_3204_BACT'],
    'Q0FF': ['nonBSA_List_S20Magnets', 'LI20_LGPS_3031_BACT'],
    'Q0D': ['nonBSA_List_S20Magnets', 'LI20_LGPS_3141_BACT'],
    'Q1D': ['nonBSA_List_S20Magnets', 'LI20_LGPS_3261_BACT'],
    'Q2D': ['nonBSA_List_S20Magnets', 'LI20_LGPS_3091_BACT']
}

bmad_quad_to_pv_map_boost = {
    # s20
    'Q1EL': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2061_BACT'],
    'Q2EL': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2131_BACT'],
    'Q3EL_1': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2151_BACT'],
    'Q3EL_2': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2151_BACT'],
    'Q4EL_1': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2201_BACT'],
    'Q4EL_2': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2201_BACT'],
    'Q4EL_3': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2201_BACT'],
    'Q5EL': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2231_BACT'],

    'Q5ER': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2262_BACT'],
    'Q4ER_1': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2281_BACT'],
    'Q4ER_2': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2281_BACT'],
    'Q4ER_3': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2281_BACT'],
    'Q3ER_1': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2341_BACT'],
    'Q3ER_2': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2341_BACT'],
    'Q2ER': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2371_BACT'],
    'Q1ER': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2441_BACT']
}

# sextupoles
bmad_sextupoles_to_pv_map = {
    'S1EL': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2145_BACT'],
    'S2EL': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2165_BACT'],
    'S3EL_1': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2195_BACT'],
    'S3EL_2': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2195_BACT'],
    'S3ER_2': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2275_BACT'],
    'S3ER_1': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2275_BACT'],
    'S2ER': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2335_BACT'],
    'S1ER': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2365_BACT']
}

# sextupole offsets (in mm) (s1l, s2l, s2r, s1r)
sextupole_offsets_x_from_daq = [['nonBSA_List_S20Magnets', 'SIOC_SYS1_ML00_AO552'],
                                ['nonBSA_List_S20Magnets', 'SIOC_SYS1_ML00_AO502'],
                                ['nonBSA_List_S20Magnets', 'SIOC_SYS1_ML00_AO517'],
                                ['nonBSA_List_S20Magnets', 'SIOC_SYS1_ML00_AO567']]

sextupole_offsets_y_from_daq = [['nonBSA_List_S20Magnets', 'SIOC_SYS1_ML00_AO557'],
                                ['nonBSA_List_S20Magnets', 'SIOC_SYS1_ML00_AO507'],
                                ['nonBSA_List_S20Magnets', 'SIOC_SYS1_ML00_AO522'],
                                ['nonBSA_List_S20Magnets', 'SIOC_SYS1_ML00_AO572']]

# injector cavities
def get_l0a_phase(database):
    """Read the L0A phase from the DAQ database and apply the FACET phase offset.
    """
    return np.mean(database._data["scalars"]["nonBSA_List_S10RF"]['KLYS_LI10_31_SFB_PDES'])-20

def get_l0a_ampl(database):
    """Read the L0A amplitude from the database and convert to Tao voltage units.
    """
    return np.mean(database._data["scalars"]["nonBSA_List_S10RF"]['KLYS_LI10_31_ADES'])*2.864664e6

def get_l0b_phase(database):
    """Read the L0B phase from the database.
    """
    return np.mean(database._data["scalars"]["nonBSA_List_S10RF"]['KLYS_LI10_41_SFB_PDES'])

def get_l0b_ampl(database):
    """Read the L0B amplitude from the database and convert to Tao voltage units.
    """
    return np.mean(database._data["scalars"]["nonBSA_List_S10RF"]['KLYS_LI10_41_ADES'])*2.864664e6

def get_l1_phase(database):
    """Read the L1 cavity phase from the database.
    """
    return np.mean(database._data["scalars"]["nonBSA_List_LINAC_KLYS"]['KLYS_LI11_11_SSSB_PDES'])

def get_l2_phase(database):
    """Read the L2 cavity phase from the database.
    """
    return np.mean(database._data["scalars"]["nonBSA_List_LINAC_KLYS"]['LI14_SBST_1_PHAS'])

def get_l3_phase(database):
    """Read the L3 cavity phase from the database.
    """
    return np.mean(database._data["scalars"]["nonBSA_List_LINAC_KLYS"]['LI19_SBST_1_PHAS'])

# L3
bmad_cavity_to_pv_map = {
    'K19_8A1': ['nonBSA_List_S20Magnets', 'LI19_KLYS_81_ADES'],
    'K19_8A2': ['nonBSA_List_S20Magnets', 'LI19_KLYS_81_ADES'],
    'K19_8A3': ['nonBSA_List_S20Magnets', 'LI19_KLYS_81_ADES']
}

bmad_corrector_to_pv_map_before_dogleg = {
    # Correctors before L0AFEND
    'YC10122': ["nonBSA_List_S10", 'YCOR_IN10_122_BDES'],
    'XC10121': ["nonBSA_List_S10", 'XCOR_IN10_121_BDES'],
    'XC10221': ["nonBSA_List_S10", 'XCOR_IN10_221_BDES'],
    'YC10222': ["nonBSA_List_S10", 'YCOR_IN10_222_BDES'],
    'YC10312': ["nonBSA_List_S10", 'YCOR_IN10_312_BDES'],
    'XC10311': ["nonBSA_List_S10", 'XCOR_IN10_311_BDES'],
    
    # Correctors after L0AFEND
    'YC10382': ["nonBSA_List_S10", 'YCOR_IN10_382_BDES'],
    'XC10381': ["nonBSA_List_S10", 'XCOR_IN10_381_BDES'],
    'YC10412': ["nonBSA_List_S10", 'YCOR_IN10_412_BDES'],
    'XC10411': ["nonBSA_List_S10", 'XCOR_IN10_411_BDES'],
    'YC10492': ["nonBSA_List_S10", 'YCOR_IN10_492_BDES'],
    'XC10491': ["nonBSA_List_S10", 'XCOR_IN10_491_BDES'],
    'XC10521': ["nonBSA_List_S10", 'XCOR_IN10_521_BDES'],
    'YC10522': ["nonBSA_List_S10", 'YCOR_IN10_522_BDES'],
    'XC10641': ["nonBSA_List_S10", 'XCOR_IN10_641_BDES'],
    'YC10642': ["nonBSA_List_S10", 'YCOR_IN10_642_BDES'],
}
bmad_corrector_to_pv_map = {    
    # Correctors after the dogleg beginning
    'XC10721': ["nonBSA_List_S10", 'XCOR_IN10_721_BDES'],
    'YC10722': ["nonBSA_List_S10", 'YCOR_IN10_722_BDES'],
    'XC10761': ["nonBSA_List_S10", 'XCOR_IN10_761_BDES'],
    'YC10762': ["nonBSA_List_S10", 'YCOR_IN10_762_BDES'],
    
    # correctors in sector 11
    'YC11105': ["nonBSA_List_S11", 'YCOR_LI11_105_BCON'],
    'XC11104': ["nonBSA_List_S11", 'XCOR_LI11_104_BCON'],
    'YC11141': ["nonBSA_List_S11", 'YCOR_LI11_141_BCON'],
    'XC11140': ["nonBSA_List_S11", 'XCOR_LI11_140_BCON'],
    'XC11202': ["nonBSA_List_S11", 'XCOR_LI11_202_BCON'],
    'YC11203': ["nonBSA_List_S11", 'YCOR_LI11_203_BCON'],
    'YC11273': ["nonBSA_List_S11", 'YCOR_LI11_273_BCON'],
    'XC11272': ["nonBSA_List_S11", 'XCOR_LI11_272_BCON'],
    'YC11305': ["nonBSA_List_S11", 'YCOR_LI11_305_BCON'],
    'XC11304': ["nonBSA_List_S11", 'XCOR_LI11_304_BCON'],
    'YC11321': ["nonBSA_List_S11", 'YCOR_LI11_321_BCON'],
    'YC11365': ["nonBSA_List_S11", 'YCOR_LI11_365_BCON'],
    'XC11398': ["nonBSA_List_S11", 'XCOR_LI11_398_BCON'],
    'YC11399': ["nonBSA_List_S11", 'YCOR_LI11_399_BCON']
}


### Lattice edit functions

def edit_tao_based_on_experiment_database(tao, dataset, correctors_coef=-1/10, correctors_from_beg=False):
    '''
    - Bend settings are disabled because setting B_FIELD in BMAD changes the positions of all downstream elements.
    - Correctors may be either all enabled (with correctors_coef != 0), enabled only from the dogleg beginning (correctors_from_beg=False), or turned off (correctors_coef=0).
    Cavities:
    - The phase set to all klystrons is the same (for a given cavity). But the EPICS databases suggest that in experiment there are two klystrons with +- large phase offset. It is disregarded here.
    - The voltage is also the same, and is tuned to match the default 125, 335, 4500, 10000 MeV.
    '''
    # Imported here (rather than at module level) to avoid a circular import with
    # simulationFunctions, which itself imports functions from this module.
    from .simulationFunctions import setAllWChicaneSextupolesXOffsets, setAllWChicaneSextupolesYOffsets

    # tao.cmd(f'set ele L0BF PHI0 = {l0bphase / 360.}')
    # tao.cmd(f'set ele L0BF VOLTAGE = {(61.0e6 + (mean_energy_MeV_lattice-125)*1e6) / math.cos(2*math.pi*l0bphase/360)}')

    tao = edit_energy_tao_based_on_experiment_database(tao, dataset)

    for k, v in bmad_quad_to_pv_map.items():
        #print(f'{k} base {np.mean(dataset._data["scalars"][v[0]][v[1]])}')
        quad_integrated_T = np.mean(dataset._data["scalars"][v[0]][v[1]])
        if k in bmad_quad_to_pv_map_boost:
            #print(f'{k} boost {np.mean(dataset._data["scalars"][bmad_quad_to_pv_map_boost[k][0]][bmad_quad_to_pv_map_boost[k][1]])}')
            quad_integrated_T += np.mean(dataset._data["scalars"][bmad_quad_to_pv_map_boost[k][0]][bmad_quad_to_pv_map_boost[k][1]])
        setQuadkG(tao, k, quad_integrated_T)

    # for k, v in bmad_bend_to_pv_map.items():
    #     bend_T = np.mean(dataset._data["scalars"][v[0]][v[1]])*tao.ele_gen_attribs(k)["ANGLE"]/tao.ele_gen_attribs(k)["L"]/0.299792458
    #     # print(f'{k} bend T from DAQ: {bend_T}')
    #     tao.cmd(f'set ele {k} B_FIELD = {bend_T}')
    #     # print(f'{k}: {tao.ele_gen_attribs(k)["B_FIELD"]}')


    for k, v in bmad_sextupoles_to_pv_map.items():
        sextupole_DAQ = np.mean(dataset._data["scalars"][v[0]][v[1]])
        setSextkG(tao, k, sextupole_DAQ)
        #print(sextupole_DAQ)
    
    sextXOffsets = np.array([0,0,0,0,0,0])
    sextYOffsets = np.array([0,0,0,0,0,0])
    for i in range(2):
        sextXOffsets[i] = np.mean(dataset._data["scalars"][sextupole_offsets_x_from_daq[i][0]][sextupole_offsets_x_from_daq[i][1]])*1e-3
        sextYOffsets[i] = np.mean(dataset._data["scalars"][sextupole_offsets_y_from_daq[i][0]][sextupole_offsets_y_from_daq[i][1]])*1e-3
    for i in range(2):
        sextXOffsets[-i] = np.mean(dataset._data["scalars"][sextupole_offsets_x_from_daq[-i][0]][sextupole_offsets_x_from_daq[-i][1]])*1e-3
        sextYOffsets[-i] = np.mean(dataset._data["scalars"][sextupole_offsets_y_from_daq[-i][0]][sextupole_offsets_y_from_daq[-i][1]])*1e-3
    setAllWChicaneSextupolesXOffsets(tao, sextXOffsets[0], sextXOffsets[1], sextXOffsets[2], sextXOffsets[3], sextXOffsets[4], sextXOffsets[5])
    setAllWChicaneSextupolesYOffsets(tao, sextYOffsets[0], sextYOffsets[1], sextYOffsets[2], sextYOffsets[3], sextYOffsets[4], sextYOffsets[5])

    if correctors_from_beg:
        for k, v in bmad_corrector_to_pv_map_before_dogleg.items():
            tao.cmd(f'set ele {k} BL_KICK = {np.mean(dataset._data["scalars"][v[0]][v[1]])*correctors_coef}')  # need 1/10 to transform kG-m (unit of PV) to T-m (units of BMAD)
    for k, v in bmad_corrector_to_pv_map.items():
        tao.cmd(f'set ele {k} BL_KICK = {np.mean(dataset._data["scalars"][v[0]][v[1]])*correctors_coef}')  # need 1/10 to transform kG-m (unit of PV) to T-m (units of BMAD)
    
    return tao

# bends
bmad_bend_to_pv_map = {
    # s10
    # NOT SAVED IN DAQ?
    # 'BCX10451': ["", ''],
    # 'BCX10461': ["", ''],
    # 'BCX10475': ["", ''],
    # 'BCX10481': ["", ''],
    'BX10661': ["nonBSA_List_S10", 'BEND_IN10_661_BDES'],
    'BX10751': ["nonBSA_List_S10", 'BEND_IN10_751_BDES'],
    # s11
    'BCX11314': ["nonBSA_List_S11", 'BEND_LI11_314_BCON'],
    'BCX11331': ["nonBSA_List_S11", 'BEND_LI11_331_BCON'],
    'BCX11338': ["nonBSA_List_S11", 'BEND_LI11_338_BCON'],
    'BCX11355': ["nonBSA_List_S11", 'BEND_LI11_355_BCON'],
    # s14
    # NOT SAVED IN DAQ?
    # 'BCX14720': ["", ''],
    # 'BCX14796': ["", ''],
    # 'BCX14808': ["", ''],
    # 'BCX14883': ["", ''],
    # s20
    'B1LE': ["nonBSA_List_S20Magnets", 'LI20_LGPS_1990_BACT'],
    'WIGE1': ["nonBSA_List_S20Magnets", 'LI20_LGPS_2420_BACT'],
    'WIGE3': ["nonBSA_List_S20Magnets", 'LI20_LGPS_2420_BACT'],
    'B1RE': ["nonBSA_List_S20Magnets", 'LI20_LGPS_1990_BACT'],
    'B2LE': ["nonBSA_List_S20Magnets", 'LI20_LGPS_2110_BACT'],
    'B3LE': ["nonBSA_List_S20Magnets", 'LI20_LGPS_2240_BACT'],
    'B3RE': ["nonBSA_List_S20Magnets", 'LI20_LGPS_2240_BACT'],
    'B2RE': ["nonBSA_List_S20Magnets", 'LI20_LGPS_2110_BACT'],
    'WIGE2': ["nonBSA_List_S20Magnets", 'LI20_BTRM_2420_BACT']
}

def get_mean_energy_of_some_dipoles(dataset, dipoles):
    return np.mean(np.array([np.mean(dataset._data["scalars"][bmad_bend_to_pv_map[dipole][0]][bmad_bend_to_pv_map[dipole][1]]) for dipole in dipoles]))

def save_dipole_energies_from_the_DAQ_database(dataset):
    bendEnergiesMeV=[125, 335, 4500, 10000]
    bendEnergiesMeV[0] = get_mean_energy_of_some_dipoles(dataset, ["BX10661", "BX10751"])*1e3
    bendEnergiesMeV[1] = get_mean_energy_of_some_dipoles(dataset, ["BCX11314", "BCX11331", "BCX11338", "BCX11355"])*1e3
    bendEnergiesMeV[2] = 4500 # not saved in DAQ, so just set to the default value
    bendEnergiesMeV[3] = get_mean_energy_of_some_dipoles(dataset, ["B1LE", "B1RE", "B2LE", "B3LE", "B3RE", "B2RE"])*1e3
    return bendEnergiesMeV


def edit_energy_tao_based_on_experiment_database(tao, dataset):
    '''
    Cavities:
    - The phase set to all klystrons is the same (for a given cavity). But the EPICS databases suggest that in experiment there are two klystrons with +- large phase offset. It is disregarded here.
    - The voltage is also the same, and is tuned to match the default 125, 335, 4500, 10000 MeV.
    '''
    l0a_phase = get_l0a_phase(dataset)
    l0a_ampl = get_l0a_ampl(dataset)
    
    l0b_phase = get_l0b_phase(dataset)
    l0b_ampl = get_l0b_ampl(dataset)

    #print([l0a_phase, l0a_ampl, l0b_phase, l0b_ampl])

    tao.cmd(f'set ele L0AF PHI0 = {l0a_phase / 360.}')
    tao.cmd(f'set ele L0AF VOLTAGE = {l0a_ampl}')
    
    tao.cmd(f'set ele L0BF PHI0 = {l0b_phase / 360.}')
    tao.cmd(f'set ele L0BF VOLTAGE = {l0b_ampl}')

    edited_energy_dogleg_MeV = 125
    current_e_start = tao.ele_gen_attribs('BEGINNING')["P0C"]
    current_e_dogleg = tao.ele_gen_attribs('BX10661')["P0C"]
    coef_e = 1 + 1e6*(edited_energy_dogleg_MeV - current_e_dogleg*1e-6)/(current_e_dogleg-current_e_start)
    tao.cmd(f'set ele L0AF VOLTAGE = {l0a_ampl*coef_e}')
    tao.cmd(f'set ele L0BF VOLTAGE = {l0b_ampl*coef_e}')

    l1_phase = get_l1_phase(dataset)
    l2_phase = get_l2_phase(dataset)
    l3_phase = get_l3_phase(dataset)
    setLinacPhase(tao, "L1", l1_phase)
    setLinacGradientAuto(tao, "L1", (335-125)*1e6)
    setLinacPhase(tao, "L2", l2_phase)
    setLinacGradientAuto(tao, "L2", (4500-335)*1e6)
    setLinacPhase(tao, "L3", l3_phase)
    setLinacGradientAuto(tao, "L3", (10000-4500)*1e6)
    
    return tao
