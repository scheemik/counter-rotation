import numpy as np
import matplotlib.pyplot as plt

omega = 1.0
kz    = 5.0
A = 1.0
B = 1.0

t0 = 0.0
tf = 10 * omega / (2*np.pi)
nt = 512
dt = (tf-t0)/nt

z0 = 0.0
zf = 1.0
nz = 256
dz = (zf-z0)/nz

t = np.linspace(t0, tf, nt)
z = np.linspace(z0, zf, nz)
tm, zm = np.meshgrid(t, z)
y = A*np.cos(omega *2*np.pi*tm - kz*zm) + B*np.cos(omega *2*np.pi*tm + kz*zm)

plt.pcolormesh(tm, zm, y)
plt.show()

#fourier transform
fy = np.fft.fft(y)
freq = np.fft.fftfreq(len(y), dt)

# fig, ax = plt.subplots(2,1)
# ax[0].plot(t, y)
# ax[0].set_xlabel("time")
# ax[1].plot(freq, fy.real, freq, fy.imag)
# ax[1].set_xlabel("frequency")
# plt.tight_layout()
# plt.show()
