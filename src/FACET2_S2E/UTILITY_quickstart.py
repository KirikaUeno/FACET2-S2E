"""Deprecated location of UTILITY_quickstart, kept so old imports keep working.

The code was split into:
    FACET2_S2E.simulation.core
    FACET2_S2E.beam.manipulation
    FACET2_S2E.beam.analysis
    FACET2_S2E.lattice.optics
    FACET2_S2E.beam.microbunching
    FACET2_S2E.simulation.config
Import from there, or use FACET2_S2E.<name> directly.
"""
import warnings

from .simulation.core import (
    initializeTao,
    applyBMADCollectiveEffectSettings,
    trackBeam,
    trackBeamHelper,
    getBeamAtElement,
    writeBeam,
    makeBeamActiveBeamFile,
)
from .beam.manipulation import (
    ballisticPropagation,
    nudgeMacroparticleWeights,
    getDriverAndWitness,
    centerBeam,
    collimateBeam,
    sortIndices,
    sliceBeam,
    getSingleBeamSlice,
    modifyAndSaveInputBeam,
)
from .beam.analysis import (
    smallestInterval,
    smallestIntervalImpliedSigma,
    smallestIntervalImpliedEmittanceModelFunction,
    smallestIntervalImpliedEmittance,
    emittance,
    getBeamSpecs,
    generalizedEmittanceSolverObjective,
    generalizedEmittanceSolver,
)
from .lattice.optics import (
    displayMatrix,
    getMatrix,
    getMatrixLEGACY,
    setLatticeAndGetMatrix,
    calcBMAG,
    launchTwissCorrectionObjective,
    launchTwissCorrection,
)
from .beam.microbunching import (
    addLHmodulation,
)
from .simulation.config import (
    loadConfig,
    disableAutoQuadEnergyCompensation,
    disableAutoMagnetEnergyCompensation,
    applyOtherConfig,
)
from .plotting.phase_space import (
    plotMod,
    slicePlotMod,
)
from .plotting.floorplan import (
    floorplanPlot,
)
from .lattice.linac import (
    getLinacMatchStrings,
    setLinacPhase,
    setLinacGradientAuto,
)
from .lattice.set_lattice import (
    setLattice,
    getBendkG,
    getQuadkG,
    getSextkG,
    setBendkG,
    setQuadkG,
    setSextkG,
    setXOffset,
    setYOffset,
    setKickerkG,
    getKickerkG,
    setBendGeVc,
    getBendGeVc,
)
from .codes.impact import (
    runImpact,
)
from .lattice.final_focus import (
    finalFocusSolver,
)
from .codes.qpad import (
    QPAD_sim,
    run_QPAD,
)

# filePathGlobal is not re-exported: it is module state of
# FACET2_S2E.simulation.core, and a copy here would be stale.
# Use tao.filePathGlobal.

warnings.warn(
    "FACET2_S2E.UTILITY_quickstart is deprecated and will be removed; "
    "import from FACET2_S2E.simulation.core, FACET2_S2E.beam.manipulation, "
    "FACET2_S2E.beam.analysis, FACET2_S2E.lattice.optics, "
    "FACET2_S2E.beam.microbunching, FACET2_S2E.simulation.config or use "
    "FACET2_S2E.<name> instead.",
    FutureWarning,
    stacklevel=2,
)
