"""
FACET2_S2E: Simulation tools for FACET-II start-to-end beam dynamics
"""

from .UTILITY_quickstart import (
    # Core initialization and tracking
    initializeTao,
    applyBMADCollectiveEffectSettings,
    trackBeam,
    trackBeamHelper,
    getBeamAtElement,
    ballisticPropagation,
    
    # Beam manipulation and analysis
    nudgeMacroparticleWeights,
    getDriverAndWitness,
    writeBeam,
    makeBeamActiveBeamFile,
    centerBeam,
    collimateBeam,
    sliceBeam,
    getSingleBeamSlice,
    
    # Statistical analysis
    smallestInterval,
    smallestIntervalImpliedSigma,
    smallestIntervalImpliedEmittance,
    smallestIntervalImpliedEmittanceModelFunction,
    emittance,
    getBeamSpecs,
    
    # Matrix and lattice operations
    displayMatrix,
    getMatrix,
    getMatrixLEGACY,
    setLatticeAndGetMatrix,
    
    # Beam modulation
    addLHmodulation,
    calcBMAG,
    
    # Configuration utilities
    loadConfig,
    
    # Optimization and correction
    launchTwissCorrection,
    launchTwissCorrectionObjective,
    generalizedEmittanceSolver,
    generalizedEmittanceSolverObjective,
    
    # Lattice configuration
    disableAutoQuadEnergyCompensation,
    disableAutoMagnetEnergyCompensation,
    applyOtherConfig,
    
    # Helper utilities
    sortIndices,
)


from .UTILITY_QPAD import (
    # QPAD visualization and analysis
    plotInteractiveQPADFigure,
    saveAllQPADFigures,
    plotPlasmaProfile,
)

# Import commonly used functions from other utility modules
from .UTILITY_plotMod import plotMod, slicePlotMod, floorplanPlot
from .UTILITY_linacPhaseAndAmplitude import getLinacMatchStrings, setLinacPhase, setLinacGradientAuto
from .UTILITY_setLattice import (
    setLattice, 
    getBendkG, getQuadkG, getSextkG,
    setBendkG, setQuadkG, setSextkG,
    setXOffset, setYOffset,
    getKickerkG, setKickerkG,
    getBendGeVc, setBendGeVc
)
from .UTILITY_finalFocusSolver import finalFocusSolver

# Functions ported over from FACET2_S2E_Kladov (kept there too, unmodified).

from .beamFunctions import (
    ## Bunch support functions

    ### Create a bunch
    make_simple_bunch,
    make_simple_bunch_flatter,
    make_simple_bunch_standalone,
    make_simple_bunch_theory_from_bunch_sims,
    ### Modify the bunch as a whole (sizes, means, chirps, correlations),
    modifyInputBeamSimple,
    #sqrtm_psd,
    #invsqrtm_psd,
    edit_bunch_parameters_from_PG,
    edit_bunch_parameters,
    ### Cut the bunch to a certain length
    cut_length,
)

from .simulationFunctions import (
    ## Initialize and run a simulation

    ### High end
    get_tao_from_experiment,
    set_beam,
    run_initialized_sim,
    run_initialized_sim_edit_bunch_energy,
    ### Correct the lattice to match the desired Pz
    tune_to_P0Cs,
    edit_energy_based_on_beam_inj,
    edit_energy_based_on_beam_L1,
    edit_energy_based_on_beam_L2,
    edit_energy_based_on_beam_L3,
    edit_energy_based_on_beam_all,

    ## Scans
    make_1d_scan,
    make_comparison_dz_2nd_order,

    ## Simulation parameters functions

    # Get lattice info
    get_element_array,
    get_rij,
    get_tijk,
    ### Sextupole settings
    setAllWChicaneSextupoles,
    setAllWChicaneSextupolesXOffsets,
    setAllWChicaneSextupolesYOffsets,
)

from .DAQdatasetToSimFunctions import (
    ## Edit lattice according to experiment

    ### BMAD to DAQ funcs
    get_l0a_phase,
    get_l0a_ampl,
    get_l0b_phase,
    get_l0b_ampl,
    ### Lattice edit functions
    edit_tao_based_on_experiment_database,
    edit_energy_tao_based_on_experiment_database,
)

from .plottingFunctions import (
    ## Plotting functions

    enable_plt_styling,
    ### Display a bunch. Aliased: FACET2_S2E already has its own plotMod (UTILITY_plotMod above);
    ### this one is Kladov's, kept available under a distinct name.
    plotMod as plotModKladov,
    print_result,
    print_result_from_tao,
    print_result_from_file,

    ## Make a nice plot from arrays
    #'normalize_arrays,
    make_a_plot,
)

from .microbunchingFunctions import (
    ### Add microbunching to a bunch
    make_modulated_bunch,

    ## Display the bunch density distribution \rho(z)
    hist,

    ## Spectrum functions
    get_nbins_for_wavelength,
    get_spectrum,
    print_spec,
    #find_nearest,
    get_spec_band,
    analyze_spec,
    get_microbunching_gain_from_beams,
    get_microbunching_gain,
)

__all__ = [
    # Core initialization and tracking
    'initializeTao',
    'applyBMADCollectiveEffectSettings',
    'trackBeam',
    'trackBeamHelper',
    'getBeamAtElement',
    'ballisticPropagation',
    
    # Beam manipulation and analysis
    'nudgeMacroparticleWeights',
    'getDriverAndWitness',
    'writeBeam',
    'makeBeamActiveBeamFile',
    'centerBeam',
    'collimateBeam',
    'sliceBeam',
    'getSingleBeamSlice',
    
    # Statistical analysis
    'smallestInterval',
    'smallestIntervalImpliedSigma',
    'smallestIntervalImpliedEmittance',
    'smallestIntervalImpliedEmittanceModelFunction',
    'emittance',
    'getBeamSpecs',
    
    # Matrix and lattice operations
    'displayMatrix',
    'getMatrix',
    'getMatrixLEGACY',
    'setLatticeAndGetMatrix',
    
    # Beam modulation
    'addLHmodulation',
    'calcBMAG',
    
    # Configuration utilities
    'loadConfig',
    
    # Optimization and correction
    'launchTwissCorrection',
    'launchTwissCorrectionObjective',
    'generalizedEmittanceSolver',
    'generalizedEmittanceSolverObjective',
    
    # Lattice configuration
    'disableAutoQuadEnergyCompensation',
    'disableAutoMagnetEnergyCompensation',
    'applyOtherConfig',
    
    # Helper utilities
    'sortIndices',
    
    # QPAD visualization and analysis
    'plotInteractiveQPADFigure',
    'saveAllQPADFigures',
    'plotPlasmaProfile',
    
    # Plotting utilities
    'plotMod',
    'slicePlotMod',
    'floorplanPlot',
    
    # Linac configuration
    'getLinacMatchStrings',
    'setLinacPhase',
    'setLinacGradientAuto',
    
    # Lattice element control
    'setLattice',
    'getBendkG', 'getQuadkG', 'getSextkG',
    'setBendkG', 'setQuadkG', 'setSextkG',
    'setXOffset', 'setYOffset',
    'getKickerkG', 'setKickerkG',
    'getBendGeVc', 'setBendGeVc',
    
    # Final focus optimization
    'finalFocusSolver',

    # Functions ported over from FACET2_S2E_Kladov (kept there too, unmodified).

    ## Bunch support functions

    ### Create a bunch
    'make_simple_bunch',
    'make_simple_bunch_flatter',
    'make_simple_bunch_standalone',
    'make_simple_bunch_theory_from_bunch_sims',
    ### Add microbunching to a bunch
    'make_modulated_bunch',
    ### Modify the bunch as a whole (sizes, means, chirps, correlations),
    'modifyInputBeamSimple',
    #'sqrtm_psd',
    #'invsqrtm_psd',
    'edit_bunch_parameters_from_PG',
    'edit_bunch_parameters',
    ### Cut the bunch to a certain length
    'cut_length',

    ## Initialize and run a simulation (Kladov's)

    ### High end
    'get_tao_from_experiment',
    'set_beam',
    'run_initialized_sim',
    'run_initialized_sim_edit_bunch_energy',
    ### Correct the lattice to match the desired Pz
    'tune_to_P0Cs',
    'edit_energy_based_on_beam_inj',
    'edit_energy_based_on_beam_L1',
    'edit_energy_based_on_beam_L2',
    'edit_energy_based_on_beam_L3',
    'edit_energy_based_on_beam_all',

    ## Edit lattice according to experiment

    ### BMAD to DAQ funcs
    'get_l0a_phase',
    'get_l0a_ampl',
    'get_l0b_phase',
    'get_l0b_ampl',
    ### Lattice edit functions
    'edit_tao_based_on_experiment_database',
    'edit_energy_tao_based_on_experiment_database',

    ## Scans

    'make_1d_scan',
    'make_comparison_dz_2nd_order',

    ## Simulation parameters functions (Kladov's)

    # Get lattice info
    'get_element_array',
    'get_rij',
    'get_tijk',
    ### Sextupole settings
    'setAllWChicaneSextupoles',
    'setAllWChicaneSextupolesXOffsets',
    'setAllWChicaneSextupolesYOffsets',

    ## Plotting functions (Kladov's)

    'enable_plt_styling',
    ### Display a bunch
    'plotModKladov',
    'print_result',
    'print_result_from_tao',
    'print_result_from_file',

    ## Make a nice plot from arrays
    #'normalize_arrays',
    'make_a_plot',

    ## Display the bunch density distribution \rho(z)
    'hist',

    ## Spectrum functions
    'get_nbins_for_wavelength',
    'get_spectrum',
    'print_spec',
    #'find_nearest',
    'get_spec_band',
    'analyze_spec',
    'get_microbunching_gain_from_beams',
    'get_microbunching_gain',
]

__version__ = '0.1.0'
