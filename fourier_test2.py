import numpy as np
import matplotlib.pyplot as plt

def make_subplot(hori, vert, data, fig, ax, h_label, v_label, title):
    im = ax.pcolormesh(hori, vert, data)
    ax.set_xlabel(h_label)
    ax.set_ylabel(v_label)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)

omega = 2.0
omega2 = omega*3
kz    = 2.0
kz2   = kz
A = 1.0
B = A * 1

# number of oscillation periods
n_o = 4
t0 = 0.0
tf = (2*np.pi) / omega * n_o
nt = 1024
dt = (tf-t0)/nt

z0 = 0.0
zf = 1.0
zf = (2*np.pi) / kz
nz = 1024
dz = (zf-z0)/nz

t = np.linspace(t0, tf, nt)
z = np.linspace(z0, zf, nz)
tm, zm = np.meshgrid(t, z)
# Total field
y = A*np.cos(omega * tm - kz*zm) + B*np.cos(omega2 * tm + kz2*zm)
# Separate up and down analytically for confirmation that CD is working
up = A*np.cos(omega * tm - kz*zm)
dn = B*np.cos(omega2 * tm + kz2*zm)

## Step 1

# fourier transform in time
fty = np.fft.fft(y, axis=1)
# find relevant frequencies
freq = np.fft.fftfreq(len(t), dt)
ftm, zft = np.meshgrid(freq, z)

# Filter out negative frequencies
for i in range(ftm.shape[0]):
    for j in range(ftm.shape[1]):
        if ftm[i][j] < 0.0:
            # Gets rid of negative f's
            fty[i][j] = 0
        else:
            # Corrects for lost amplitude
            fty[i][j] = fty[i][j] * 2.0

# inverse fourier transform in time
ifty = np.fft.ifft(fty, axis=1)
#   ify is now a complex valued signal where ifty.real == y, or at least close enough

### Step 2

# fourier transform in spatial dimension (z)
#   similar to ft in time, but switch dimensions around
fzyp = np.fft.fft(ifty, axis=0)
# find relevant wavenumbers
wavn = np.fft.fftfreq(len(z), dz)
tfz, fzm = np.meshgrid(t, wavn)
# make a copy for the negative wave numbers
fzyn = fzyp.copy()

# Filter out one half of wavenumbers to separate up and down
for i in range(fzm.shape[0]):
    for j in range(fzm.shape[1]):
        if fzm[i][j] > 0.0:
            # for up, remove values for positive wave numbers
            fzyp[i][j] = 0.0
        else:
            # for down, remove values for negative wave numbers
            fzyn[i][j] = 0.0

# inverse fourier transform in space (z)
ifzyp = np.fft.ifft(fzyp, axis=0)
ifzyn = np.fft.ifft(fzyn, axis=0)

# Get up and down fields as F = |mag_f| * exp(i*phi_f)
up_field = ifzyp.real * np.exp(np.real(1j * ifzyp.imag))
dn_field = ifzyn.real * np.exp(np.real(1j * ifzyn.imag))

# plotting
diff = False
if diff == True:
    rows = 1
    cols = 3
    fig, ax = plt.subplots(rows, cols)
    make_subplot(tm, zm, y-(up_field+dn_field), fig, ax[0], r'$t$', r'$z$', r'Total difference')
    make_subplot(tm, zm, up-up_field, fig, ax[1], r'$t$', r'$z$', r'Up difference')
    make_subplot(tm, zm, dn-dn_field, fig, ax[2], r'$t$', r'$z$', r'Down difference')
else:
    rows = 2
    cols = 3
    fig, ax = plt.subplots(rows, cols)
    make_subplot(tm, zm, y, fig, ax[0][0], r'$t$', r'$z$', r'$A\cos(\omega t-k_z z)+B\cos(\omega t+k_z z)$')
    make_subplot(tm, zm, up, fig, ax[0][1], r'$t$', r'$z$', r'$A\cos(\omega t-k_z z)$')
    make_subplot(tm, zm, dn, fig, ax[0][2], r'$t$', r'$z$', r'$B\cos(\omega t+k_z z)$')
    make_subplot(tm, zm, up_field+dn_field, fig, ax[1][0], r'$t$', r'$z$', 'Up + Down')
    make_subplot(tm, zm, up_field, fig, ax[1][1], r'$t$', r'$z$', 'Up')
    make_subplot(tm, zm, dn_field, fig, ax[1][2], r'$t$', r'$z$', 'Down')

# Set aspect ratio for figure
size_factor = 3.0
w, h = cols*size_factor, (rows+0.4)*size_factor
plt.gcf().set_size_inches(w, h)
plt.tight_layout()
plt.show()
