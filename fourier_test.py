import numpy as np
import matplotlib.pyplot as plt

def make_subplot(hori, vert, data, fig, ax, h_label, v_label, title):
    #im = ax.pcolormesh(data)
    im = ax.pcolormesh(hori, vert, data)
    ax.set_xlabel(h_label)
    ax.set_ylabel(v_label)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)

omega = 2.0
kz    = 2.0
A = 1.0
B = 1.0

number_of_periods = 5
t0 = 0.0
tf = (2*np.pi) / omega * number_of_periods
nt = 512
dt = (tf-t0)/nt

z0 = 0.0
zf = 1.0
zf = (2*np.pi) / kz
nz = 256
dz = (zf-z0)/nz

t = np.linspace(t0, tf, nt)
z = np.linspace(z0, zf, nz)
tm, zm = np.meshgrid(t, z)
y = A*np.cos(omega * tm - kz*zm) + B*np.cos(omega * tm + kz*zm)
print('y ', y.shape)
print('tm ', tm.shape)
print('zm ', zm.shape)
up = A*np.cos(omega * tm - kz*zm)
dn = B*np.cos(omega * tm + kz*zm)

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
print('ifty ', ifty.shape)
#   ify is now a complex valued signal where ifty.real == y, or at least close enough

### Step 2

# fourier transform in spatial dimension (z)
#   similar to ft in time, but switch dimensions around
fzyp = np.fft.fft(ifty, axis=0)
print('fzyp ', fzyp.shape)
# find relevant wavenumbers
wavn = np.fft.fftfreq(len(z), dz)
print('wavn ', wavn.shape)
print('t ', t.shape)
tfz, fzm = np.meshgrid(t, wavn)
print('fzm ', fzm.shape)
print(fzm)
# make a copy for the negative wave numbers
fzyn = fzyp.copy()

# Filter out one half of wavenumbers to separate up and down
for i in range(fzm.shape[0]):
    for j in range(fzm.shape[1]):
        if fzm[i][j] > 0.0:
            # for up, double values for positive wave numbers
            fzyp[i][j] = fzyp[i][j] * 2.0
            # for down, remove values for positive wave numbers
            fzyn[i][j] = 0.0
        else:
            # for up, remove values for negative wave numbers
            fzyp[i][j] = 0.0
            # for down, double values for negative wave numbers
            fzyn[i][j] = fzyn[i][j] * 2.0

# inverse fourier transform in space (z)
ifzyp = np.fft.ifft(fzyp, axis=0)
ifzyn = np.fft.ifft(fzyn, axis=0)

# Get up and down fields as F = |mag_f| * exp(i*phi_f)
up_field = ifzyp.real * np.exp(np.real(1j * ifzyp.imag))
dn_field = ifzyn.real * np.exp(np.real(1j * ifzyn.imag))

# plotting
fig, ax = plt.subplots(3,1)
make_subplot(tm, zm, y, fig, ax[0], r'$t$', r'$z$', 'OG')
make_subplot(tm, zm, up_field, fig, ax[1], r'$t$', r'$z$', 'up')
make_subplot(tm, zm, dn_field, fig, ax[2], r'$t$', r'$z$', 'down')

plt.tight_layout()
plt.show()
