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
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
#plt.ioff()

from docopt import docopt

args = docopt(__doc__)
h5_files = args['<files>']
import pathlib
output_path = pathlib.Path(args['--output']).absolute()

## add block to check if 'frames' exists and to create it if not
#   make sure it doesn't require dedalus libraries, then delete block at bottom

dpi = 100
tasks = ['b', 'w']
cmap = 'RdBu_r'

###############################################################################
# Helper functions

# Saves figure as a frame
def save_fig_as_frame(fig, index, output, dpi):
    savename = 'write_{:06}.png'.format(index)
    savepath = output.joinpath(savename)
    fig.savefig(str(savepath), dpi=dpi)
    fig.clear()

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

#for i in range(len(t_axis)):
for i in range(2):
    fig, ax0 = plt.subplots(nrows=1)
    ax0.pcolormesh(x_axis, z_axis, dsets[0][35][1], cmap=cmap)
    ax0.set_aspect(AR)
    save_fig_as_frame(fig, i, output_path, dpi)


###############################################################################
###############################################################################

if __name__ == "__main__":

    import pathlib
    from docopt import docopt
    from dedalus.tools import logging
    from dedalus.tools import post
    from dedalus.tools.parallel import Sync

    args = docopt(__doc__)
    h5_files = args['<files>']
    output_path = pathlib.Path(args['--output']).absolute()
    # Create output directory if needed
    with Sync() as sync:
        if sync.comm.rank == 0:
            if not output_path.exists():
                output_path.mkdir()
    #post.visit_writes(args['<files>'], main, output=output_path)
