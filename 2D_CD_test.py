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
    # Filter out negative frequencies (and other freq's maybe? Band pass anyone?)
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

# fourier transform in spatial dimensions
#   similar to FT in time, but switch dimensions around
def FT_in_space(t, x, z, data, dx, dz):
    # FT in space (x) of the data (axis 1 is x) for positive k_x
    AB = np.fft.fft(data, axis=1)
    # make a copy for the negative k_x
    CD = AB.copy()
    # find relevant wavenumbers
    k_xs = np.fft.fftfreq(len(x), dx)
    t_grid, kx_grid, z_grid = np.meshgrid(t, k_xs, z, indexing='ij')
    # Filter out half the wavenumbers to separate positive and negative
    for i in range(kx_grid.shape[0]):
        for j in range(kx_grid.shape[1]):
            for l in range(kx_grid.shape[2]):
                if kx_grid[i][j][l] > 0.0:
                    # for AB, remove values for positive wave numbers
                    AB[i][j][l] = 0.0
                else:
                    # for CD, remove values for negative wave numbers
                    CD[i][j][l] = 0.0
    # inverse fourier transform in space (x)
    kx_p = np.fft.ifft(AB, axis=1)
    kx_n = np.fft.ifft(CD, axis=1)
    ##
    # FT in space (z) of the data (axis 2 is z) for positive k_z
    A = np.fft.fft(kx_p, axis=2)
    C = np.fft.fft(kx_n, axis=2)
    # make copies for the negative k_x
    B = A.copy()
    D = C.copy()
    # find relevant wavenumbers
    k_zs = np.fft.fftfreq(len(z), dz)
    t_grid, x_grid, kz_grid = np.meshgrid(t, x, k_zs, indexing='ij')
    # Filter out half the wavenumbers to separate positive and negative
    for i in range(kz_grid.shape[0]):
        for j in range(kz_grid.shape[1]):
            for l in range(kz_grid.shape[2]):
                if kz_grid[i][j][l] > 0.0:
                    # for A and C, remove values for positive wave numbers
                    A[i][j][l] = 0.0
                    C[i][j][l] = 0.0
                else:
                    # for B and D, remove values for negative wave numbers
                    B[i][j][l] = 0.0
                    D[i][j][l] = 0.0
    # inverse fourier transform in space (x)
    A_xp_zp = np.fft.ifft(A, axis=2)
    B_xp_zn = np.fft.ifft(B, axis=2)
    C_xn_zp = np.fft.ifft(C, axis=2)
    D_xn_zn = np.fft.ifft(D, axis=2)
    return A_xp_zp, B_xp_zn, C_xn_zp, D_xn_zn

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
nt = 256
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
y = Af + Bf

## Step 1
print('taking FT in time')
ift_t_y = FT_in_time(t, x, z, y, dt)

## Step 2
print('taking FT in space')
A_cd, B_cd, C_cd, D_cd = FT_in_space(t, x, z, ift_t_y, dx, dz)

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
        plot_t_slice(xm[i], zm[i], y[i].real, fig, ax[0][0], r'$x$', r'$z$', r'OG')
        plot_t_slice(xm[i], zm[i], A_cd[i].real+B_cd[i].real, fig, ax[0][1], r'$x$', r'$z$', r'2D CD')
        plot_t_slice(xm[i], zm[i], A_cd[i].real, fig, ax[1][0], r'$x$', r'$z$', r'A CD')
        plot_t_slice(xm[i], zm[i], B_cd[i].real, fig, ax[1][1], r'$x$', r'$z$', r'B CD')
        # plot_t_slice(xm[i], zm[i], A_cd[i].real, fig, ax[0][0], r'$x$', r'$z$', r'A')
        # plot_t_slice(xm[i], zm[i], B_cd[i].real, fig, ax[0][1], r'$x$', r'$z$', r'B')
        # plot_t_slice(xm[i], zm[i], C_cd[i].real, fig, ax[1][0], r'$x$', r'$z$', r'C')
        # plot_t_slice(xm[i], zm[i], D_cd[i].real, fig, ax[1][1], r'$x$', r'$z$', r'D')
        fig.tight_layout() # this (mostly) prevents axis labels from overlapping
        # Save figure as image in designated output directory
        save_fig_as_frame(fig, i, output_path, dpi)
        plt.close(fig)

# then run
# python3 create_gif.py 2D_CD_test 2D_CD_test.gif 2D_CD_frames
