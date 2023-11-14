#%%
import numpy as np 
import matplotlib.pyplot as plt 
#%%
# Constants

# Constants
hbar = 1.0  # Planck's constant / 2pi
m = 1.0     # Particle mass
L = 100.0   # Total length of the region
dx = 0.1    # Spatial step size
dt = 0.001   # Time step size
N = 1000#int(L / dx) + 1  # Number of spatial points
T = 100.0   # Total simulation time

# Potential barrier parameters
V0 = 2.0       # Barrier height
a = 0.1        # Barrier width
barrier_start = 50.0 - a/2.0  # Barrier starting position
barrier_end = 50.0 + a/2.0    # Barrier ending position

# Wavefunction parameters
x0 = 45.0     # Center of the wave packet
sigma = 5.0   # Width of the wave packet
q = 1.0       # Wavevector

# Initialization
x = np.linspace(0, L, N)    # Spatial grid
psi = np.sqrt(1/(np.sqrt(np.pi) * sigma))**(-1/4) * np.exp(1j * q * (x - x0)) * np.exp(-(x - x0)**2 / (2 * sigma**2))

# Potential barrier
V = np.zeros(N)
V[(x >= barrier_start) & (x <= barrier_end)] = V0

# Numerical evolution
steps = int(T / dt)
for n in range(steps):
    psi_new = np.zeros(N, dtype=complex)
    for i in range(1, N-1):
        psi_new[i] = psi[i] + (1j * hbar * dt / (2 * m * dx**2)) * (psi[i+1] - 2*psi[i] + psi[i-1]) - (1j * dt / hbar) * V[i] * psi[i]

    psi = psi_new

# Plotting the result
plt.plot(x, np.abs(psi)**2)
plt.xlabel('Position')
plt.ylabel('Probability Density')
plt.show()
# %%
