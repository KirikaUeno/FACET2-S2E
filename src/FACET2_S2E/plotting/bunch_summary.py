"""Display a bunch: phase-space panels plus printed moments.

``plotMod`` here is exported from the package as ``plotModKladov`` to avoid
clashing with plotting.phase_space.plotMod.

Moved from plottingFunctions.py.
"""

from scipy.stats import moment
from copy import copy
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import numpy as np
import pmd_beamphysics.labels
import pmd_beamphysics.units
from pmd_beamphysics import ParticleGroup

from ..beam.manipulation import cut_length
from ..simulation.core import getBeamAtElement


### Display a bunch

# New PlotMod for horizontal plotting

def plotMod(particle_group, key1='t', key2='p', 
                  bins=None,
                  *,
                  xlim=None,
                  ylim=None,
                  tex=True,
                  nice=True,
            fig=None, outer=None, i=None, z_from_t=False,
                  **kwargs):
    """
    Derived from openPMD-beamphysics marginal_plot()
    """    

    #plt.close('all')
    
    CMAP0 = copy(plt.get_cmap('viridis'))
    CMAP0.set_under(CMAP0(0))  # set under-color to the lowest colormap color
    CMAP1 = copy(plt.get_cmap('plasma'))

    # Suppress display while the sub-axes are composed; restored before returning, otherwise
    # every later figure in the session silently stops being shown.
    was_interactive = plt.isinteractive()
    plt.ioff()
    
    if not bins:
        n = len(particle_group)
        bins = int(np.sqrt(n/4) )

    key1changed = False
    key2changed = False
    if z_from_t:
        if key1=='z':
            key1='delta_t'
            key1changed = True
        if key2=='z':
            key2='delta_t'
            key2changed = True

    # Scale to nice units and get the factor, unit prefix
    x = particle_group[key1]
    y = particle_group[key2]

    if key1changed:
        x = -3e8*x
    if key2changed:
        y = -3e8*x
    
    # Form nice arrays
    x, f1, p1, xmin, xmax = pmd_beamphysics.units.plottable_array(x, nice=nice, lim=xlim)
    y, f2, p2, ymin, ymax = pmd_beamphysics.units.plottable_array(y, nice=nice, lim=ylim)

    if key1changed:
        x = f1*x
    if key2changed:
        y = f2*x
    
    w = particle_group['weight']
    
    u1 = particle_group.units(key1).unitSymbol
    u2 = particle_group.units(key2).unitSymbol
    ux = p1+u1
    uy = p2+u2
    
    labelx = pmd_beamphysics.labels.mathlabel(key1, units=ux, tex=tex)
    labely = pmd_beamphysics.labels.mathlabel(key2, units=uy, tex=tex)

    if key1changed:
        labelx = "z (m)"
    if key2changed:
        labely = "z"

    if (fig is None or outer is None or i is None):
        fig = plt.figure(**kwargs)
        gs = GridSpec(4,4)
        ax_joint = fig.add_subplot(gs[1:4,0:3])
        ax_marg_x = fig.add_subplot(gs[0,0:3])
        ax_marg_y = fig.add_subplot(gs[1:4,3])
    else:
        gs = GridSpecFromSubplotSpec(
            4, 4,
            subplot_spec=outer[i],
            wspace=0.0,
            hspace=0.0
        )
        
    ax_joint = fig.add_subplot(gs[1:4, 0:3])
    ax_marg_x = fig.add_subplot(gs[0, 0:3], sharex=ax_joint)
    ax_marg_y = fig.add_subplot(gs[1:4, 3], sharey=ax_joint)

    # Set the joint plot background color to match the bottom end of the colormap
    #ax_joint.set_facecolor(CMAP0(0))
    ax_joint.set_facecolor('white')
    
    # Plot the hexbin
    ax_joint.hexbin(x, y, C=w, reduce_C_function=np.sum, gridsize=bins, cmap=CMAP0, vmin=1e-20)
    
    # Top histogram
    hist, bin_edges = np.histogram(x, bins=bins, weights=w)
    hist_x = bin_edges[:-1] + np.diff(bin_edges) / 2
    hist_width =  np.diff(bin_edges)
    hist_y, hist_f, hist_prefix = pmd_beamphysics.units.nice_array(hist/hist_width)
    ax_marg_x.bar(hist_x, hist_y, hist_width, color='gray')
    if u1 == 's':
        _, hist_prefix = pmd_beamphysics.units.nice_scale_prefix(hist_f/f1)
        ax_marg_x.set_ylabel(f'{hist_prefix}A')
    else:   
        ax_marg_x.set_ylabel(pmd_beamphysics.labels.mathlabel(f'{hist_prefix}C/{ux}'))

    # Side histogram
    hist, bin_edges = np.histogram(y, bins=bins, weights=w)
    hist_x = bin_edges[:-1] + np.diff(bin_edges) / 2
    hist_width =  np.diff(bin_edges)
    hist_y, hist_f, hist_prefix = pmd_beamphysics.units.nice_array(hist/hist_width)
    ax_marg_y.barh(hist_x, hist_y, hist_width, color='gray')
    ax_marg_y.set_xlabel(pmd_beamphysics.labels.mathlabel(f'{hist_prefix}C/{uy}'))

    # Turn off tick labels on marginals
    plt.setp(ax_marg_x.get_xticklabels(), visible=False)
    plt.setp(ax_marg_y.get_yticklabels(), visible=False)
    
    # Set labels on joint
    ax_joint.set_xlabel(labelx)
    ax_joint.set_ylabel(labely)
    
    if xlim:
        ax_joint.set_xlim(xmin/f1, xmax/f1)      
        ax_marg_x.set_xlim(xmin/f1, xmax/f1)
        
    if ylim:
        ax_joint.set_ylim(ymin/f2, ymax/f2)     
        ax_marg_y.set_ylim(ymin/f2, ymax/f2)

    if was_interactive:
        plt.ion()
    
    return ax_joint, ax_marg_x, ax_marg_y


# Prints specified 2d spaces, plots are arranged horizontally. Uncomment a section for vertical plotting.
def print_result(particle_group, length = 0, couples = [['t','pz']], moments = True, sliceEnergyMoment=False, energyInDelta=False, drift_to_z_in_cutting = True, z_from_t=False):
    """Display 2D phase-space plots and print basic bunch moments for a ParticleGroup."""
    P = particle_group.copy()
    #P.drift_to_z()
    if length != 0:
        Pt = cut_length(P, length, drift_to_z = drift_to_z_in_cutting)
    else:
        Pt = P

    l = len(couples)
    fig = plt.figure(figsize=(7*l,6))
    
    # outer layout: 1 row, l columns
    outer = GridSpec(1, l, wspace=0.3)
    
    for (i,couple) in enumerate(couples):
        plotMod(Pt, couple[0], couple[1], bins=300, fig=fig, outer=outer, i=i, z_from_t=z_from_t)

    plt.show()

    # if column-arranging is needed
    # for (i,couple) in enumerate(couples):
    #     #display(plotMod(Pt, couple[0], couple[1],  bins=300))
    #     plt.subplot(1,len(couples[:,1]),i+1)
    #     plotMod(Pt, couple[0], couple[1],  bins=300)
    #     plt.show()
        
    if moments:
        deltas = (P.gamma-np.mean(P.gamma))/np.mean(P.gamma)
        energies = P.pz
        energies = P.energy
        if sliceEnergyMoment:
            Pslice = cut_length(P, length = 1e-7)
            deltas = (Pslice.gamma-np.mean(Pslice.gamma))/np.mean(Pslice.gamma)
            energies = Pslice.pz
        if energyInDelta:
            sigmapz = moment(deltas, moment=2) ** 0.5
        else:
            sigmapz = moment(energies, moment=2) ** 0.5
        print([float(moment(Pt.x, moment=2) ** 0.5), float(moment(Pt.xp, moment=2) ** 0.5), float(moment(Pt.y, moment=2) ** 0.5), float(moment(Pt.yp, moment=2) ** 0.5), float(moment(Pt.z, moment=2) ** 0.5), float(sigmapz), float(moment(Pt.t, moment=2) ** 0.5)])


def print_result_from_tao(tao_local, location, length = 0, couples = [['t','pz']], moments = True, sliceEnergyMoment=False, energyInDelta=False, z_from_t=False):
    """Display results (2d distribution plots) for a beam extracted from Tao at a given location."""
    print_result(getBeamAtElement(tao_local, location, tToZ=False), length = length, couples = couples, moments = moments, sliceEnergyMoment=sliceEnergyMoment, energyInDelta=energyInDelta, z_from_t=z_from_t)


def print_result_from_file(file, length = 0, couples = [['t','pz']], moments = True, sliceEnergyMoment=False, energyInDelta=False, z_from_t=False):
    """Display results (2d distribution plots) for a beam loaded from an HDF5 file."""
    print_result(ParticleGroup(file), length = length, couples = couples, moments = moments, sliceEnergyMoment=sliceEnergyMoment, energyInDelta=energyInDelta, z_from_t=z_from_t)
