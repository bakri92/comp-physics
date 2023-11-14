#%% 
import numpy as np
import matplotlib.pyplot as plt
#%%
# 1: random walk 
# Num of steps
N = 1000
N_particles = 10000


def random_walk(N,N_particles):
    """
    random walk function that gives back the end point
    starting point is always 0, added to the array
    so array legnth of single walk is N+1

    Parameters
    ----------
    N : int
        Numbers of steps
    N_particles : int
        Number of praticles
    start_x : int, optional
        Starting point , by default 0

    Returns
    -------
    NArray
        array of size (N_particles,N+1) of end points of walk for all N
    """
    zeros = np.zeros(N_particles)
    moving_dir = np.array([1,-1])
    random_move = np.random.choice(moving_dir,[N_particles,N],p=[0.5,0.5])
    x_end = np.cumsum(random_move,axis=1)
    x_end = np.column_stack((zeros, x_end))
    return x_end

x = random_walk(N,N_particles)
x_squared = x**2
cum_squared_mean = np.mean(x_squared,axis=0)#np.sum(x_squared,axis=0) /np.arange(1,N_particles+1)
cum_mean_squared =  np.mean(x,axis=0)**2 #(np.sum(x,axis=0) / np.arange(1,N_particles+1)) ** 2

variance = cum_squared_mean - cum_mean_squared

#%%
## plots

plt.plot(np.arange(0,N+1),variance,"o",c='k',markersize=4,label="simulation")
plt.plot(np.arange(0,N+1),np.arange(0,N+1)*1,c="red",label="analytical")
plt.title("variance")
plt.xlabel('N')
plt.ylabel(r'$<x^2>-<x>^2$')
plt.legend()

#%%
## rms 

rms = np.sqrt(cum_squared_mean)
plt.plot(np.arange(0,N+1),rms,"o",c="k",markersize=4,label="simulation")
plt.plot(np.arange(0,N+1),np.sqrt(np.arange(0,N+1)),c='red',label='analytical')
plt.title("root mean square")
plt.xlabel('N')
plt.ylabel(r'$\sqrt{<sx^2>}$')
plt.legend()
print(f"After 1000 jumps from the simulation required grind length is: {mse[-1]}")
print(fr"Analytically this value is N\delta x: {np.round(np.sqrt(N))}")
