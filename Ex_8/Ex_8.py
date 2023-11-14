"""
Exercise 8: Schrodinger equation 
quantum tunneling
"""

#%% 
import numpy as np 
import matplotlib.pyplot as plt 
from scipy import sparse
from scipy.linalg import expm
#%%
L = 1001 
tau = 0.001 
m = 50000
delta = 0.1 
x_0 = 0 
x_end = 100
x0_wave = 20 
xfin_wave = 70 
a = tau /(4*delta**2)
sigma = 3

## potential
def V(x): 
    if x >= 50 and x <= 50.5:
        return 2 
    else: 
        return 0 

## starting wave packet
def omega_0(x,x0):
    A =  (2 * np.pi * sigma**2) ** (-1/4) 
    wave_function = A * np.exp(1j * (x-x0))*np.exp(-((x-x0)/(2*sigma))**2)
    return wave_function

omega = np.zeros((m,L),dtype=complex)
omega[0] = omega_0(x,x0_wave)
omega_prob = np.zeros((m,L),dtype=complex)
omega_prob[0] = np.conj(omega[0]) * omega[0]

x = np.linspace(x_0,x_end, L)

potential = np.array([V(xs) for xs in x])
        


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

H_V = expm(np.diag(potential) * -1j * tau)

## product formula 
exp_Hamiltonian = H_k1.dot(H_k2).dot(H_V).dot(H_k2).dot(H_k1)
#%%

## run the operator to integrate 
for ts in range(1,m):
    omega[ts] = exp_Hamiltonian.dot(omega[ts-1])
    omega_prob[ts] = np.conj(omega[ts]) * omega[ts]



######## PLOTS #################

## ALL TOGETHER
plt.figure(figsize=(9,5))
t_to_plot = [0, 10, 20, 25, 30, 40, 49.9]

for ts in t_to_plot: 
    plt.plot(x,omega_prob[int(ts/tau)], label=f't = {ts}')

plt.plot(x,potential, label='Potential Barrier', color = 'k')
plt.legend()
plt.title(r"Different times for for probability P(x,t)=$|\phi(t)|^2$ of a quantum particle penetrating a potential well")
plt.grid()
plt.ylim(0,0.22)

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
