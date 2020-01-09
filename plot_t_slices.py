"""
Plot planes from joint analysis files.

Usage:
    plot_slices.py <files>... [--output=<dir>]

Options:
    --output=<dir>  # Output directory [default: ./frames]

"""

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import ticker
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.ioff()

# Parse input parameters
from docopt import docopt
args = docopt(__doc__)
h5_files = args['<files>']
import pathlib
output_path = pathlib.Path(args['--output']).absolute()

# Check if output path exists, if not, create it
import os
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Labels
hori_label = r'$x$ (m)'
vert_label = r'$z$ (m)'

# Parameters
dpi = 100
tasks = ['b', 'p', 'u', 'w']
rows = 2
cols = 2
cmap = 'RdBu_r'
n_ticks = 5
n_cb_ticks = 3
round_to_decimal = 1
title_size = 'medium'
suptitle_size = 'large'

###############################################################################
# Helper functions

# Make a plot for one task
def plot_task(fig, axes, time_i, task_j, x_ax, z_ax, dsets, cmap, AR):
    ax = axes[task_j]
    print(ax)
    # plot task colormesh
    im = ax.pcolormesh(x_ax, z_ax, dsets[task_j][time_i][1], cmap=cmap)
    # format axis labels and ticks
    format_labels_and_ticks(ax)
    # format colorbar
    divider = make_axes_locatable(ax) # this is the only way I found to put the colorbar on top of the plot
    cax = divider.append_axes('top', size='5%', pad=0.03)
    fig.colorbar(im, cax=cax, orientation='horizontal', ticks=ticker.MaxNLocator(nbins=n_cb_ticks), format=ticker.FuncFormatter(latex_exp))
    cax.xaxis.set_ticks_position('top')
    # Set aspect ratio for plot
    ax.set_aspect(AR)
    # add title
    cax.set_title(tasks[task_j], fontsize=title_size)

def format_labels_and_ticks(ax):
    # add labels
    ax.set_xlabel(hori_label)
    ax.set_ylabel(vert_label)
    # fix vertical and horizontal ticks
    x0, xf = ax.get_xlim()
    x0 = round(x0, round_to_decimal)
    xf = round(xf, round_to_decimal)
    ax.xaxis.set_ticks(np.linspace(x0, xf, n_ticks))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(latex_exp))
    z0, zf = ax.get_ylim()
    z0 = round(z0, round_to_decimal)
    zf = round(zf, round_to_decimal)
    ax.yaxis.set_ticks(np.linspace(z0, zf, n_ticks))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(latex_exp))

# Saves figure as a frame
def save_fig_as_frame(fig, index, output, dpi):
    savename = 'write_{:06}.png'.format(index)
    savepath = output.joinpath(savename)
    fig.savefig(str(savepath), dpi=dpi)
    fig.clear()

# Takes an exponential number and returns a string formatted nicely for latex
#   Expects numbers in the format 7.0E+2
def latex_exp(num, pos=None):
    if (isinstance(num, int)):
        # integer type, don't reformat
        return num
    else:
        float_str = "{:.1E}".format(num)
        if "E" in float_str:
            base, exponent = float_str.split("E")
            exp = int(exponent)
            b   = float(base)
            str1 = '$'
            if (exp == -1):
                str1 = str1 + str(b/10.0)
            elif (exp == 0):
                str1 = str1 + str(base)
            elif (exp == 1):
                str1 = str1 + str(b*10.0)
            elif (exp == 2):
                str1 = str1 + str(b*100.0)
            else:
                str1 = str1 + str(base) + r'\cdot10^{' + str(exp) + '}'
            str1 = str1 + '$'
            return r"{0}".format(str1)
        else:
            return float_str

###############################################################################

# dsets will be an array containing all the data
#   it will have a size of: tasks x timesteps x 2 x nx x nz (5D)
dsets = []
for task in tasks:
    task_tseries = []
    for filename in h5_files:
        print(filename)
        with h5py.File(filename, mode='r') as f:
            dset = f['tasks'][task]
            # Check dimensionality of data
            if len(dset.shape) != 3:
                raise ValueError("This only works for 3D datasets")
            # The [()] syntax returns all data from an h5 object
            task_grid = np.array(dset[()])
            x_scale = f['scales']['x']['1.0']
            x_axis = np.array(x_scale[()])
            z_scale = f['scales']['z']['1.0']
            z_axis = np.array(z_scale[()])
            t_scale = f['scales']['sim_time']
            t_axis = np.array(t_scale[()])
            for i in range(len(t_axis)):
                time_slice = [t_axis[i], np.transpose(task_grid[i])]
                task_tseries.append(time_slice)
    dsets.append(task_tseries)

# Calculate aspect ratio of plot based on extent
extent_aspect = abs((x_axis[-1]-x_axis[0])/(z_axis[-1]-z_axis[0]))
aspect_ratio = 1
AR = extent_aspect/aspect_ratio

# Iterate across time, plotting and saving a frame for each timestep
#for i in range(len(t_axis)):
for i in range(10,12):
    fig, ax = plt.subplots(nrows=rows, ncols=cols)
    # Plot each task
    for j in range(ax.size):
        plot_task(fig, ax, i, j, x_axis, z_axis, dsets, cmap, AR)
    # Add title for overall figure
    fig.suptitle('supertitle', fontsize=suptitle_size)
    fig.tight_layout() # this (mostly) prevents axis labels from overlapping
    # Save figure as image in designated output directory
    save_fig_as_frame(fig, i, output_path, dpi)


###############################################################################
###############################################################################

# if __name__ == "__main__":
#
#     import pathlib
#     from docopt import docopt
#     from dedalus.tools import logging
#     from dedalus.tools import post
#     from dedalus.tools.parallel import Sync
#
#     args = docopt(__doc__)
#     h5_files = args['<files>']
#     output_path = pathlib.Path(args['--output']).absolute()
#     # Create output directory if needed
#     with Sync() as sync:
#         if sync.comm.rank == 0:
#             if not output_path.exists():
#                 output_path.mkdir()
#     #post.visit_writes(args['<files>'], main, output=output_path)
