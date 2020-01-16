"""
Testing out complex demodulation in 2 spatial dimensions
Based on Mercier et al. 2008

Written by Mikhail Schee
Jan 2020
"""

import numpy as np
#import math
import matplotlib.pyplot as plt
#from matplotlib.animation import FuncAnimation
#from sympy import *
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

# fourier transform in time, filter negative freq's, inverse fourier transform
def FT_in_time(t, x, z, data, dt):
    # FT in time of the data (axis 0 is time)
    ftd = np.fft.fft(data, axis=0)
    # find relevant frequencies
    freq = np.fft.fftfreq(len(t), dt)
    f_grid, x_grid, z_grid = np.meshgrid(freq, x, z, indexing='ij')
    # Filter out negative frequencies
    for i in range(f_grid.shape[0]):
        for j in range(f_grid.shape[1]):
            for l in range(f_grid.shape[2]):
                if f_grid[i][j][l] < 0.0:
                    # Gets rid of negative freq's
                    ftd[i][j][l] = 0
                else:
                    # Corrects for lost amplitude
                    ftd[i][j][l] = ftd[i][j][l] * 2.0
    # inverse fourier transform in time of the data
    iftd = np.fft.ifft(ftd, axis=0)
    #   a complex valued signal where iftd.real == data, or close enough
    return iftd

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
n_o = 4
t0 = 0.0
tf = (2*np.pi) / omega * n_o
nt = 128
dt = (tf-t0)/nt

# number of horizontal wavelengths
n_lambda_x = 2
x0 = 0.0
xf = 1.0
xf = (2*np.pi) / kx * n_lambda_x
nx = 256
dx = (xf-x0)/nx

# number of vertical wavelengths
n_lambda_z = 3
z0 = 0.0
zf = 1.0
zf = (2*np.pi) / kz * n_lambda_z
nz = 256
dz = (zf-z0)/nz

t = np.linspace(t0, tf, nt)
x = np.linspace(x0, xf, nx)
z = np.linspace(z0, zf, nz)
# using indexing='ij' insures the input order is the output order for dimensions
tm, xm, zm = np.meshgrid(t, x, z, indexing='ij')
# individual fields, following Merier et al. 2008 eq 10-13
Af = A*np.exp(1j*(omega*tm - kx*xm - kz*zm))
Bf = B*np.exp(1j*(omega*tm - kx*xm + kz*zm))
Cf = C*np.exp(1j*(omega*tm + kx*xm - kz*zm))
Df = D*np.exp(1j*(omega*tm + kx*xm + kz*zm))
# up and down fields
up = Af + Cf
dn = Bf + Df
# total field
y = up #+ dn

## Step 1
print('taking FT in time')
ift_t_y = FT_in_time(t, x, z, y, dt)

###############################################################################

plot_frames = True

if plot_frames == True:
    print('Now plotting frames')
    # Iterate across time, plotting and saving a frame for each timestep
    for i in range(len(t)):
        fig, ax = plt.subplots(nrows=2, ncols=2)
        # Set aspect ratio for figure
        # size_factor = 4.0
        # w, h = cols*size_factor, (rows+0.4)*size_factor
        # plt.gcf().set_size_inches(w, h)
        # Add title for overall figure
        title_str = '{:}, $t=${:2.2f}'
        fig.suptitle(title_str.format(name, t[i]))
        # Plot
        plot_t_slice(xm[i], zm[i], Af[i].real, fig, ax[0][0], r'$x$', r'$z$', r'A')
        plot_t_slice(xm[i], zm[i], Bf[i].real, fig, ax[0][1], r'$x$', r'$z$', r'B')
        plot_t_slice(xm[i], zm[i], Cf[i].real, fig, ax[1][0], r'$x$', r'$z$', r'C')
        plot_t_slice(xm[i], zm[i], Df[i].real, fig, ax[1][1], r'$x$', r'$z$', r'D')
        fig.tight_layout() # this (mostly) prevents axis labels from overlapping
        # Save figure as image in designated output directory
        save_fig_as_frame(fig, i, output_path, dpi)
        plt.close(fig)

# then run
# python3 create_gif.py 2D_CD_test 2D_CD_test.gif 2D_CD_frames
