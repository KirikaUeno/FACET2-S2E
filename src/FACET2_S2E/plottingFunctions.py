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

"""Plotting helpers for FACET2-S2E simulation workflows.

This module contains matplotlib styling, phase-space bunch display
(plotMod, print_result and friends), and generic publication-quality
line-plot helpers (make_a_plot).

Split out of the former functionsForSims.py.
"""

from .beamFunctions import cut_length

## Plotting functions

def enable_plt_styling():
    """Enable APS/PRAB-like matplotlib styling for plots.
    """
    # APS / PRAB-like matplotlib configuration
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "axes.linewidth": 1.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
    })


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


## Make a nice plot from arrays

def normalize_arrays(arrays):
    """Normalize input arrays into a list of numpy arrays for plotting utilities."""
    # Case 1: single NumPy array
    if isinstance(arrays, np.ndarray):
        if arrays.ndim == 1:
            return [arrays]          # wrap single dataset
        elif arrays.ndim == 2:
            return list(arrays)      # split into rows
        else:
            raise ValueError("Array must be 1D or 2D")

    # Case 2: iterable of arrays (list/tuple/etc.)
    try:
        return [np.asarray(a) for a in arrays]
    except TypeError:
        raise ValueError("Input must be an array or iterable of arrays")

def make_a_plot(x, y, errs=None, aspect_ratio=0.67, label=r"$\sin(x)$", x_label=r"$x$", y_label=r"$y$", cartesian_axes=[True, True], axes_location=[0, 0], colors=['blue', 'black', 'red']):
    """Create a publication-quality line plot from x/y arrays with optional error bands."""
    # plt.rcParams.update({
    #     "font.family": "serif",
    #     "font.size": 10,
    #     "axes.labelsize": 10,
    #     "axes.titlesize": 10,
    #     "xtick.labelsize": 9,
    #     "ytick.labelsize": 9,
    #     "legend.fontsize": 9,
    #     "axes.linewidth": 0.8,
    #     "xtick.direction": "in",
    #     "ytick.direction": "in",
    #     "xtick.major.size": 4,
    #     "ytick.major.size": 4,
    #     "xtick.minor.size": 2,
    #     "ytick.minor.size": 2,
    #     "xtick.top": True,
    #     "ytick.right": True,
    #     "figure.dpi": 300,
    #     "savefig.dpi": 300,
    # })

    x = normalize_arrays(x)
    y = normalize_arrays(y)
    if errs is not None:
        errs = normalize_arrays(errs)
    nLines = len(x)
    if nLines==1:
        label=[label]
        
    # if len(y_label)==2:
    #     fig, ax1 = plt.subplots(figsize=(3.4, 3.4*aspect_ratio))

    #     ax1.plot(x[0], y[0], lw=1.8, label=label[0], c=colors[0]) if errs is None else ax1.errorbar(x[0], y[0], errs[0], lw=1.8, label=label[0], c=colors[0])
    #     ax1.set_xlabel(x_label)
    #     ax1.set_ylabel(y_label[0])
        
    #     ax2 = ax1.twinx()
    #     ax2.plot(x[1], y[1], lw=1.8, label=label[1], c=colors[1]) if errs is None else ax2.errorbar(x[1], y[1], errs[1], lw=1.8, label=label[1], c=colors[1])
    #     ax2.set_xlabel(x_label)
    #     ax2.set_ylabel(y_label[1])
    
    #     if cartesian_axes[1]:
    #         ax2.axhline(axes_location[1], linestyle="--", linewidth=1.0, color="0.6", zorder=0)
    #     if cartesian_axes[0]:
    #         ax2.axvline(axes_location[0], linestyle="--", linewidth=1.0, color="0.6", zorder=0)

        
    #     # plt.setp(ax1.get_xticklabels(), fontsize=14)
    #     # plt.setp(ax2.get_yticklabels(), fontsize=14)
    #     # plt.setp(ax1.get_yticklabels(), fontsize=14)
        
    #     # Minor ticks
    #     ax2.xaxis.set_minor_locator(AutoMinorLocator())
    #     ax2.yaxis.set_minor_locator(AutoMinorLocator())
    #     # Tick parameters
    #     ax2.tick_params(which="both", width=1)
    #     ax2.tick_params(which="major", length=6)
    #     ax2.tick_params(which="minor", length=3)
        
    #     h1, lab1 = ax1.get_legend_handles_labels()
    #     h2, lab2 = ax2.get_legend_handles_labels()
    #     ax1.legend(h1 + h2, lab1 + lab2, loc='best', frameon=False,handlelength=2.0)

    if len(y_label) in [2, 3]:
    
        fig, ax1 = plt.subplots(
            figsize=(4.5, 4.5 * aspect_ratio)
        )
    
        # -------------------------
        # First Y axis
        # -------------------------
        if errs is None:
            ax1.plot(
                x[0], y[0],
                lw=1.8,
                label=label[0],
                c=colors[0]
            )
        else:
            ax1.errorbar(
                x[0], y[0],
                errs[0],
                lw=1.8,
                label=label[0],
                c=colors[0]
            )
    
        ax1.set_xlabel(x_label)
        ax1.set_ylabel(y_label[0])
    
        # -------------------------
        # Second Y axis
        # -------------------------
        ax2 = ax1.twinx()
    
        if errs is None:
            ax2.plot(
                x[1], y[1],
                lw=1.8,
                label=label[1],
                c=colors[1]
            )
        else:
            ax2.errorbar(
                x[1], y[1],
                errs[1],
                lw=1.8,
                label=label[1],
                c=colors[1]
            )
    
        ax2.set_ylabel(y_label[1])
    
        # -------------------------
        # Third Y axis
        # -------------------------
        if len(y_label) == 3:
    
            ax3 = ax1.twinx()
    
            # Move third axis outward
            ax3.spines["right"].set_position(("outward", 50))
    
            # Make the third spine visible
            ax3.spines["right"].set_visible(True)
    
            if errs is None:
                ax3.plot(
                    x[2], y[2],
                    lw=1.8,
                    label=label[2],
                    c=colors[2]
                )
            else:
                ax3.errorbar(
                    x[2], y[2],
                    errs[2],
                    lw=1.8,
                    label=label[2],
                    c=colors[2]
                )
    
            ax3.set_ylabel(y_label[2])
    
        # -------------------------
        # Cartesian axes
        # -------------------------
        if cartesian_axes[1]:
            ax1.axhline(
                axes_location[1],
                linestyle="--",
                linewidth=1.0,
                color="0.6",
                zorder=0
            )
    
        if cartesian_axes[0]:
            ax1.axvline(
                axes_location[0],
                linestyle="--",
                linewidth=1.0,
                color="0.6",
                zorder=0
            )
    
        # -------------------------
        # Minor ticks
        # -------------------------
        ax1.xaxis.set_minor_locator(AutoMinorLocator())
        ax1.yaxis.set_minor_locator(AutoMinorLocator())
        ax2.yaxis.set_minor_locator(AutoMinorLocator())
    
        if len(y_label) == 3:
            ax3.yaxis.set_minor_locator(AutoMinorLocator())
    
        # -------------------------
        # Tick parameters
        # -------------------------
        ax1.tick_params(which="both", width=1)
        ax1.tick_params(which="major", length=6)
        ax1.tick_params(which="minor", length=3)
    
        ax2.tick_params(which="both", width=1)
        ax2.tick_params(which="major", length=6)
        ax2.tick_params(which="minor", length=3)
    
        if len(y_label) == 3:
            ax3.tick_params(which="both", width=1)
            ax3.tick_params(which="major", length=6)
            ax3.tick_params(which="minor", length=3)

        # Axis 1
        ax1.spines["left"].set_color(colors[0])
        ax1.tick_params(axis="y", colors=colors[0])
        ax1.yaxis.label.set_color(colors[0])

        # Axis 2
        ax2.spines["right"].set_color(colors[1])
        ax2.tick_params(axis="y", colors=colors[1])
        ax2.yaxis.label.set_color(colors[1])

        # Axis 3
        if len(y_label) == 3:
            ax3.spines["right"].set_color(colors[2])
            ax3.tick_params(axis="y", colors=colors[2])
            ax3.yaxis.label.set_color(colors[2])
    
        # -------------------------
        # Combined legend
        # -------------------------
        h1, lab1 = ax1.get_legend_handles_labels()
        h2, lab2 = ax2.get_legend_handles_labels()
    
        handles = h1 + h2
        labels = lab1 + lab2
    
        if len(y_label) == 3:
            h3, lab3 = ax3.get_legend_handles_labels()
            handles += h3
            labels += lab3
    
        ax1.legend(
            handles,
            labels,
            loc="best",
            frameon=False,
            handlelength=2.0
        )

    else:
        fig, ax = plt.subplots(figsize=(3.4, 2.6))  # JACoW one-column width
        #fig, ax = plt.subplots(figsize=(6.0, 6.0*aspect_ratio))
        for i, a in enumerate(x):
            ax.plot(1000*x[i], y[i], lw=1.8, label=label[i]) if errs is None else ax.errorbar(x[i], y[i], errs[i], lw=1.8, label=label[i])
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
    
        if cartesian_axes[1]:
            ax.axhline(axes_location[1], linestyle="--", linewidth=1.0, color="0.6", zorder=0)
        if cartesian_axes[0]:
            ax.axvline(axes_location[0], linestyle="--", linewidth=1.0, color="0.6", zorder=0)
        
        # Minor ticks
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        # Tick parameters
        ax.tick_params(which="both", width=1)
        ax.tick_params(which="major", length=6)
        ax.tick_params(which="minor", length=3)
        # Legend
        #loc="upper right",
        ax.legend(loc="best",frameon=False,handlelength=2.0)
        
    # Tight layout for journal export
    fig.tight_layout(pad=0.3)
    
    # Save (recommended formats for journals)
    # fig.savefig("figure.pdf")
    # fig.savefig("figure.eps")
