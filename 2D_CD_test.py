"""
A script for visualizing by boundary forcing functions

"""

# !pip3 install --upgrade
# !pip3 install sympy
# !pip3 install matplotlib

# %%
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sympy import *
import sys

def plot_t_slice(hori, vert, data, fig, ax, h_label, v_label, title):
    im = ax.pcolormesh(hori, vert, data, shading='gouraud')
    ax.set_xlabel(h_label)
    ax.set_ylabel(v_label)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    return im

# Saves figure as a frame
def save_fig_as_frame(fig, index, output, dpi):
    savename = 'write_{:06}.png'.format(index)
    savepath = output + '/' + savename
    fig.savefig(str(savepath), dpi=dpi)
    fig.clear()

###############################################################################
# %%

output_path = '2D_CD_frames'
dpi = 100
name = 'plot test'

# Parameters
omega = 2.0
kx    = 2.0
kz   = 2.0
A = 1.0
B = A * 1
C = A * 1
D = A * 1

# number of oscillation periods
n_o = 1
t0 = 0.0
tf = (2*np.pi) / omega * n_o
nt = 128
dt = (tf-t0)/nt

x0 = 0.0
xf = 1.0
xf = (2*np.pi) / kx
nx = 256
dx = (xf-x0)/nx

z0 = 0.0
zf = 1.0
zf = (2*np.pi) / kz
nz = 512
dz = (zf-z0)/nz

t = np.linspace(t0, tf, nt)
x = np.linspace(x0, xf, nx)
z = np.linspace(z0, zf, nz)
# using indexing='ij' insures the input order is the output order for dimensions
tm, xm, zm = np.meshgrid(t, x, z, indexing='ij')
# Total field
y = A*np.exp(1j*(omega*tm - kx*xm - kz*zm))

# Iterate across time, plotting and saving a frame for each timestep
for i in range(len(t)):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    # Set aspect ratio for figure
    # size_factor = 4.0
    # w, h = cols*size_factor, (rows+0.4)*size_factor
    # plt.gcf().set_size_inches(w, h)
    # Add title for overall figure
    title_str = '{:}, $t=${:2.2f}'
    ax.set_title(title_str.format(name, t[i]))
    # Plot
    plot_t_slice(xm[i], zm[i], y[i].real, fig, ax, r'$x$', r'$z$', title_str.format(name, t[i]))
    fig.tight_layout() # this (mostly) prevents axis labels from overlapping
    # Save figure as image in designated output directory
    save_fig_as_frame(fig, i, output_path, dpi)
    plt.close(fig)

# then run
# python3 create_gif.py 2D_CD_test 2D_CD_test.gif 2D_CD_frames

###############################################################################
sys.exit("stopping script")
fig, ax = plt.subplots()
cmesh = plot_t_slice(xm[0], zm[0], y[0].real, fig, ax, r'$x$', r'$z$', r'title')

def animate(i):
    #line0.set_ydata(win[i, :])
    cmesh.set_array(y[i].real.ravel())

anim = FuncAnimation(
    fig, animate, interval=tf, frames=len(t)-1)

name = '2D_CD_test'
#filename = '_boundary_forcing/' + name + '.mp4'
#anim.save(filename)
filename = name + '.gif'
anim.save(filename, dpi=600, writer='imagemagik')

#plt.draw()
#plt.show()

# %%
