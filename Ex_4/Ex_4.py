#%% 
import numpy as np
import matplotlib.pyplot as plt

#----------------------------------------------------------------------
##  BLOCK OF FUNCTIONS USED IN THE MAIN CODE
#----------------------------------------------------------------------
def initialstate(N,size):   
    ''' generates a random spin configuration for initial condition'''
    if size == 1:
        state = 2*np.random.randint(2, size=N)-1
    elif size == 2:
        state = 2*np.random.randint(2, size=N*N)-1
    else:
        raise ValueError("size must be 1 or 2")
    return state


def ising_energy_1d(spin_chain):
    """
    Calculates the energy of a 1D Ising model spin chain.

    Parameters:
    - spin_chain (np.array): A 1D numpy array containing the spin states of the chain.

    Returns:
    - energy (float): The energy of the spin chain.
    """
    energy = -np.sum(spin_chain[:-1] * spin_chain[1:]) - spin_chain[0] * spin_chain[-1]
    return energy

def metropolis_1d(spin_chain, beta):
    """
    Performs a single step of the Metropolis Monte Carlo algorithm for 1d ising model.

    Parameters:
    - spin_chain (np.array): A 1D numpy array containing the spin states of the chain.
    - beta (float): The inverse temperature.

    Returns:
    - new_spin_chain (np.array): A 1D numpy array containing the updated spin states of the chain.
    """
    N = len(spin_chain)
    i = np.random.randint(N)
    delta_E = 2 * spin_chain[i] * (spin_chain[(i+1)%N] + spin_chain[(i-1)%N])
    if delta_E < 0 or np.exp(-beta * delta_E) > np.random.rand():
        spin_chain[i] *= -1
    return spin_chain

def metropolis_2d(spin_chain, L,T, mag, energy):
    """
    Performs a single step of the Metropolis Monte Carlo algorithm for 2d ising model.

    Parameters:
    - spin_chain (np.array): A 2D numpy array containing the spin states of the chain.
    - L (integer): the size of the grid
    - T (float): The temperature.

    Returns:
    - new_spin_chain (np.array): A 2D numpy array containing the updated spin states of the chain.
    """
    i = np.random.randint(L)
    j = np.random.randint(L)   
    dE = delta_E_2d(spin_chain, i, j)
    if dE <= 0 or np.exp(-dE/T) > np.random.rand():
            spin_chain[i, j] = -spin_chain[i, j]
            energy += dE
            mag += 2*spin_chain[i, j]    
    return spin_chain, mag, energy

def simulate_ising_model_2d(N, temp, steps):
    """
    Simulates a 1D Ising model spin chain using the Metropolis Monte Carlo algorithm.

    Parameters:
    - N (int): Length of the spin chain.
    - temp (float): Temperature of the system.
    - steps (int): Number of simulation steps to perform.

    Returns:
    - spin_chain (np.array): A 1D numpy array containing the final spin states of the chain.
    - energy_history (np.array): A 1D numpy array containing the energy of the chain at each simulation step.
    """
    spin_chain = initialstate(N, 1)
    beta = 1 / temp
    energy_history = np.zeros(steps)
    for i in range(steps):
        spin_chain = metropolis_1d(spin_chain, beta)
        energy_history[i] = ising_energy_1d(spin_chain)
    return spin_chain, energy_history


def simulate_ising_model_1d(N, temp, steps):
    """
    Simulates a 1D Ising model spin chain using the Metropolis Monte Carlo algorithm.

    Parameters:
    - N (int): Length of the spin chain.
    - temp (float): Temperature of the system.
    - steps (int): Number of simulation steps to perform.

    Returns:
    - spin_chain (np.array): A 1D numpy array containing the final spin states of the chain.
    - energy_history (np.array): A 1D numpy array containing the energy of the chain at each simulation step.
    """
    spin_chain = initialstate(N, 1)
    beta = 1 / temp
    energy_history = np.zeros(steps)
    for i in range(steps):
        spin_chain = metropolis_1d(spin_chain, beta)
        energy_history[i] = ising_energy_1d(spin_chain)
    return spin_chain, energy_history






def delta_E_2d(s, i, j, L, J):
    """calculates dE in 2D model

    Parameters
    ----------
    s : np.array
        spin chain
    i : integer
        index
    j : integer
        index
    L : float
        lattice size
    J : float
        exchange interaction parameter

    Returns
    -------
    float
        delta E
    """
    neighbors = s[(i-1)%L, j] + s[(i+1)%L, j] + s[i, (j-1)%L] + s[i, (j+1)%L]
    return 2*J*s[i, j]*neighbors