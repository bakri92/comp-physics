#%%
"""
Exam exercise: 
Quantum Harmonic Oscillator 
"""
import numpy as np 
import matplotlib.pyplot as plt 
from scipy import sparse
from scipy.linalg import expm

#%%
L = 1001 
tau = 0.00025 
m = 40000
delta = 0.025 
x_0 = -15
x_end = 15
xfin_wave = 70 
a = tau /(4*delta**2)
sigma = 1
x0_wave = 0
config1 = (1,2,0) # Omega, sigma, x0

## potential
def V(x, omega): 
    return (omega**2/2) * x ** 2


## phi_0 wave packet
def phi_0(x,x0, sigma = config1[1]):
    A =  (np.pi * sigma**2) ** (-1/4) 
    wave_function = A * np.exp(-((x-x0)/(np.sqrt(2)*sigma))**2)
    return wave_function

x = np.linspace(x_0,x_end, L)

phi = np.zeros((m,L),dtype=complex)
phi[0] = phi_0(x,config1[2])
phi_prob = np.zeros((m,L),dtype=complex)
phi_prob[0] = np.conj(phi[0]) * phi[0]


#%%
#############################
## Hamiltonian exp(-i tau H)
c_array_1 = np.ones(L,dtype=complex)
c_array_1[:L] = np.cos(a)

c_array_2 = np.ones(L,dtype=complex)
c_array_2[1:L] = np.cos(a)

s_array_1 = np.zeros(L-1,dtype=complex)
s_array_1[0:L-1:2] =  np.sin(a) 
s_array_1 = s_array_1 * 1j

s_array_2 = np.zeros(L-1,dtype=complex)
s_array_2[1:L-1:2] =  np.sin(a) 
s_array_2 = s_array_2 * 1j


H_k1 = np.zeros((L,L),dtype=complex)
np.fill_diagonal(H_k1,c_array_1)
np.fill_diagonal(H_k1[:,1:],s_array_1)
np.fill_diagonal(H_k1[1:,:],s_array_1)

H_k2 = np.zeros((L,L),dtype=complex)
np.fill_diagonal(H_k2,c_array_2)
np.fill_diagonal(H_k2[:,1:],s_array_2)
np.fill_diagonal(H_k2[1:,:],s_array_2)

H_V = expm(np.diag((delta**-2)+potential) * -1j * tau)

## product formula 
exp_Hamiltonian = H_k1.dot(H_k2).dot(H_V).dot(H_k2).dot(H_k1)
#%%

## run the operator to integrate 
for ts in range(1,m):
    phi[ts] = exp_Hamiltonian.dot(phi[ts-1])
    phi_prob[ts] = np.conj(phi[ts]) * phi[ts]

#%%

######## PLOTS #################

## ALL TOGETHER
plt.figure(figsize=(9,5))
t_to_plot = np.arange(0,10,0.5)

for ts in t_to_plot: 
    plt.plot(x,phi_prob[int(ts/tau)], label=f't = {ts}')

#plt.plot(x,potential, label='Potential Barrier', color = 'k')
plt.legend()
plt.title(r"Different times for for probability P(x,t)=$|\phi(t)|^2$ of a quantum particle penetrating a potential well")
plt.grid()
plt.ylim(0,0.8)

#%%
## EACH TIME STEP 
for ts in t_to_plot: 
    plt.figure(figsize=(8,4))
    plt.plot(x,omega_prob[int(ts/tau)], label=f't = {ts}',c='red')
    plt.plot(x,potential, label='Potential Barrier', color = 'k',alpha=0.6)
    plt.legend()
    plt.grid()
    plt.title(rf"t={ts} for P(x,t)=$|\phi(x,t)|^2$ of a quantum particle penetrating a potential barrier")
    plt.ylim(0,0.22)
    plt.show()



# %%
### Integrating for x >= 50.5
snapshot_reduced =np.array([omega_prob[0][506:], omega_prob[int(10/tau)][506:], omega_prob[int(20/tau)][506:], omega_prob[int(25/tau)][506:], omega_prob[int(30/tau)][506:], omega_prob[int(40/tau)][506:], omega_prob[int(49.9/tau)][506:]])

snapshot_reduced_intg = np.trapz(snapshot_reduced,x=x[506:],dx=0.01, axis=1)

snapshot_reduced_intg_tot = np.sum(snapshot_reduced_intg)
snapshot_reduced_intg_norm = snapshot_reduced_intg / snapshot_reduced_intg_tot
snapshot_reduced_intg / np.max(snapshot_reduced_intg)
