#%%
import numpy as np 
import matplotlib.pyplot as plt 
#%% 
## exercise 1  

# solving d2x/dt2=-x assuming m=k=1
# solved with two first degree pdes:
# x(t + Δt) = x(t) + v(t) * Δt
# v(t + Δt) = v(t) - x(t) * Δt

# Parameters
#
x0 = 1.0  # Initial position
v0 = 0.0  # Initial velocity
dt = np.array([0.1,0.01,0.001])  # Time step

t_01 = np.arange(1,10000+0.1,0.1)
t_001 = np.arange(1,10000+0.01,0.01)
t_0001 = np.arange(1,10000+0.001,0.001)
ts = [t_01,t_001,t_0001]
# %%
xs = []
vs = []
ts = []
for i,delta in enumerate(dt):
    num_steps = int(10000/delta)
    x = np.zeros(num_steps)
    v = np.zeros(num_steps)
    t = np.zeros(num_steps)
    v[0] = 1 
    for j in range(1,num_steps):
        t[j] = j * delta
        x[j] = x[j-1] + v[j-1] * delta
        v[j] = v[j-1] - x[j-1] * delta
    xs.append(x)
    vs.append(v)
    ts.append(t)
# %%
plt.plot(ts[0],xs[0],label="simulation")
plt.plot(ts[0],np.sin(ts[0]),'--',label="analytical sin(t)")
plt.xlim(0,200)
plt.ylim(-2,2)
plt.legend()
#plt.xlim(0,40)
plt.grid()
plt.title(r"Harmonic Oscillator with one particle using Euler algorithm for $\Delta t=0.1$")
plt.xlabel("t")
plt.ylabel("x(t)")
plt.savefig("ex_11.png")

#%%
plt.plot(ts[1],xs[1],label="simulation")
plt.plot(ts[1],np.sin(ts[1]),'--',label="analytical sin(t)")
plt.ylim(-2,2)
plt.xlim(0,200)
plt.grid()
plt.title(r"Harmonic Oscillator with one particle using Euler algorithm for $\Delta t=0.01$")
plt.xlabel("t")
plt.ylabel("x(t)")
plt.savefig("ex_12.png")
#%%
plt.plot(ts[2],xs[2],label="simulation")
plt.plot(ts[2],np.sin(ts[2]),'--',label="analytical sin(t)")
plt.ylim(-2,2)
plt.xlim(0,400)
plt.grid()
plt.title(r"Harmonic Oscillator with one particle using Euler algorithm for $\Delta t=0.001$")
plt.xlabel("t")
plt.ylabel("x(t)")
plt.savefig("ex_13.png")

#%%
######################################################
######################################################
######################################################
## exercise 2: 



xs_a = []
vs_a = []
ts_a = []

xs_b = []
vs_b = []
ts_b = []
for i,delta in enumerate(dt):
    num_steps = int(10000/delta)
    x = np.zeros(num_steps)
    v = np.zeros(num_steps)
    t = np.zeros(num_steps)
    v[0] = 1 
    ## implementation a 
    for j in range(1, num_steps):
        t[j] = j * delta
        v[j] = v[j-1] - x[j-1] * delta
        x[j] = x[j-1] + v[j] * delta
    xs_a.append(x)
    vs_a.append(v)
    ts_a.append(t)
    x = np.zeros(num_steps)
    v = np.zeros(num_steps)
    t = np.zeros(num_steps)
    v[0] = 1 
    ## implementation b
    for j in range(1, num_steps):
        t[j] = j * delta
        x[j] = x[j-1] + v[j-1] * delta
        v[j] = v[j-1] - x[j] * delta
    xs_b.append(x)
    vs_b.append(v)
    ts_b.append(t)

# %%
plt.plot(ts_a[1][0:1000],xs_a[1][0:1000],label="x(t)")
plt.plot(ts_a[1][0:1000],vs_a[1][0:1000],label="v(t)")
plt.plot(ts_a[1][0:1000],(vs_a[1][0:1000])**2/2+(xs_a[1][0:1000])**2/2,label="E(t)")
plt.plot(ts_a[1][0:1000],np.sin(ts_a[1][0:1000]),'--',label="sin(t)")
plt.legend()
plt.grid()
plt.title(r"Harmonic Oscillator with one particle using first Euler-Cormer algorithm for $\Delta t=0.01$")
plt.xlabel("t")
plt.ylabel("x(t),v(t)")
plt.savefig("ex_21.png")
#%%
plt.plot(ts_b[1][0:1000],xs_b[1][0:1000],label="x(t)")
plt.plot(ts_b[1][0:1000],vs_b[1][0:1000],label="v(t)")
plt.plot(ts_b[1][0:1000],(vs_b[1][0:1000])**2/2+(xs_b[1][0:1000])**2/2,label="E(t)")
plt.plot(ts_b[1][0:1000],np.sin(ts_b[1][0:1000]),'--',label="sin(t)")
plt.legend()
plt.grid()
plt.title(r"Harmonic Oscillator with one particle using second Euler-Cormer algorithm for $\Delta t=0.01$")
plt.xlabel("t")
plt.ylabel("x(t),v(t)")
plt.savefig("ex_22.png")
#plt.ylim(0.45,0.55)
#plt.ylim(-2,2)
# %%
#############################################################
#############################################################
#############################################################

## exercise 3 

xs = []
vs = []
ts = []
for i,delta in enumerate(dt):
    num_steps = int(10000/delta)
    x = np.zeros(num_steps)
    v = np.zeros(num_steps)
    t = np.zeros(num_steps)
    v[0] = 1 
    for j in range(1,num_steps):
        t[j] = j * delta
        ## velocity_verlet
        # pos is updated with current velocity and previous position
        # in addition to acceleration term
        x[j] = x[j-1] + v[j-1] * delta + 0.5 * (-x[j-1]) * delta**2
        # velocity is updated using the current and next positions
        v[j] = v[j-1] + 0.5 * (-x[j]) * delta + 0.5 * (-x[j-1]) * delta
    xs.append(x)
    vs.append(v)
    ts.append(t)
# %%
plt.plot(ts[1][0:1000],xs[1][0:1000],label="x(t)")
plt.plot(ts[1][0:1000],vs[1][0:1000],label="v(t)")
plt.plot(ts[1][0:1000],(vs[1][0:1000])**2/2+(xs[1][0:1000])**2/2,label="E(t)")
plt.plot(ts[1][0:1000],np.sin(ts[1][0:1000]),'--',label="sin(t)")
plt.legend()
plt.grid()
plt.title(r"Harmonic Oscillator with one particle using velocity Verlet algorithm for $\Delta t=0.01$")
plt.xlabel("t")
plt.ylabel("x(t),v(t)")
plt.savefig("ex_31.png")
# %%
#######################################################
#######################################################
#######################################################
 
## exercise 4: 


def acc(k:int,N:int,x:np.array): 
    """return acceleration 
    Parameters
    ----------
    k : int
        particle
    N : int
        total number of particles
    x : np.array
        pos at current time step
    Returns
    -------
    float
        acceleration at the given particle k 
    """
    if k==0: 
        return -(x[k] - x[k+1])
    elif k==N-1:
        return -(x[k] - x[k-1])
    else: 
        return -(2*x[k] - x[k-1] - x[k+1])

def E(x:np.array,v:np.array):
    """calculating the energy according to the hamiltonian

    Parameters
    ----------
    x : np.array
        array of position
    v : np.array
        array of velocity
    Returns
    -------
    np.array
        energy array at each time step

    """
    first_term = np.sum(v ** 2,axis =1)/2
    xs = []
    for i in range(0,x.shape[1]-1):
        a = (x[:,i] - x[:,i+1])**2
        xs.append(a)
    xs_sum = np.sum(xs,axis=0)
    #print(xs)
    second_term = xs_sum/2
    return first_term + second_term
    
Ns = [4,16,128]
x0 = 1
dt = [0.01]
xs = []
vs = []
x_N = []
v_N = []
for N in Ns:
    for i,delta in enumerate(dt):
        num_steps = int(100/delta)
        x = np.zeros((num_steps+1,N))
        x[0,int(N/2)-1] = x0
        v = np.zeros((num_steps+1,N))
        t = np.zeros(num_steps+1)
        for j in range(1,num_steps+1):
            t[j] = j * delta
            # pos is updated with current velocity and previous position
            # in addition to acceleration term
            prev_acc = np.zeros(N)
            current_acc = np.zeros(N)
            ## update acceleration of previous step
            for n in range(N):
                prev_acc[n] = acc(n,N,x[j-1,:]) 
            ## update position 
            x[j,:] = x[j-1,:] + (v[j-1,:] * delta) + (0.5 * prev_acc * delta**2)
            ## update acceleration of current step
            for n in range(N):
                current_acc[n] = acc(n,N,x[j,:]) 
            ## update velocity 
            v[j,:] = v[j-1,:] +( 0.5 * delta * prev_acc) +(0.5 * delta * current_acc)
    x_N.append(x)
    v_N.append(v)


#%%
for i, xN in enumerate(x_N):
    new_x = xN.T
    for j, x in enumerate(new_x):
        plt.plot(t,x,label=f"$N_{j+1}$")
    plt.plot(t,E(xN,v_N[i]),label="E(t)")
    plt.title(fr"Coupled Harmonic Oscillator for {len(new_x)} Particles with initial config 1 and $\Delta t=0.01$")
    plt.ylabel("x(t)")
    plt.xlabel("t")
    plt.ylim(-2,2)
    plt.xlim(0,10)
    if len(new_x) == 4:
        plt.legend()
    plt.grid()
    plt.show()
    #plt.savefig(f"ex_31_{i}.png",bbox_inches="tight")

#%%
Ns = [4,16,128]
dt = [0.01]
xs = []
vs = []
x_N = []
v_N = []
for N in Ns:
    for i,delta in enumerate(dt):
        num_steps = int(100/delta)
        x = np.zeros((num_steps+1,N))
        for n in range(N):
            x[0,n] = np.sin(np.pi*n/(N+1))
        v = np.zeros((num_steps+1,N))
        t = np.zeros(num_steps+1)
        for j in range(1,num_steps+1):
            t[j] = j * delta
            # pos is updated with current velocity and previous position
            # in addition to acceleration term
            prev_acc = np.zeros(N)
            current_acc = np.zeros(N)
            ## update acceleration of previous step
            for n in range(N):
                prev_acc[n] = acc(n,N,x[j-1,:]) 
            ## update position 
            x[j,:] = x[j-1,:] + (v[j-1,:] * delta) + (0.5 * prev_acc * delta**2)
            ## update acceleration of current step
            for n in range(N):
                current_acc[n] = acc(n,N,x[j,:]) 
            ## update velocity 
            v[j,:] = v[j-1,:] +( 0.5 * delta * prev_acc) +(0.5 * delta * current_acc)
    x_N.append(x)
    v_N.append(v)
#%%
for i, xN in enumerate(x_N):
    new_x = xN.T
    for j, x in enumerate(new_x):
        plt.plot(t,x,label=f"$N_{j+1}$")
    plt.plot(t,E(xN,v_N[i]),label="E(t)")
    plt.title(f"Coupled Harmonic Oscillator for {len(new_x)} Particles with initial config 2 and j=1")
    plt.ylabel("x(t)")
    plt.xlabel("t")
   # plt.ylim(-2,2)
   # plt.xlim(0,10)
    plt.grid()
    if len(new_x) == 4:
        plt.legend()
    plt.savefig(f"ex_32_{i+1}.png",bbox_inches="tight")
    plt.show()

# %%
Ns = [4,16,128]
dt = [0.01]
xs = []
vs = []
x_N = []
v_N = []
for N in Ns:
    for i,delta in enumerate(dt):
        num_steps = int(100/delta)
        x = np.zeros((num_steps+1,N))
        for n in range(N):
            x[0,n] = np.sin((np.pi*n*(N/2))/(N+1))
        v = np.zeros((num_steps+1,N))
        t = np.zeros(num_steps+1)
        for j in range(1,num_steps+1):
            t[j] = j * delta
            # pos is updated with current velocity and previous position
            # in addition to acceleration term
            prev_acc = np.zeros(N)
            current_acc = np.zeros(N)
            ## update acceleration of previous step
            for n in range(N):
                prev_acc[n] = acc(n,N,x[j-1,:]) 
            ## update position 
            x[j,:] = x[j-1,:] + (v[j-1,:] * delta) + (0.5 * prev_acc * delta**2)
            ## update acceleration of current step
            for n in range(N):
                current_acc[n] = acc(n,N,x[j,:]) 
            ## update velocity 
            v[j,:] = v[j-1,:] +( 0.5 * delta * prev_acc) +(0.5 * delta * current_acc)
    x_N.append(x)
    v_N.append(v)
#%%
for i, xN in enumerate(x_N):
    new_x = xN.T
    for j, x in enumerate(new_x):
        plt.plot(t,x,label=f"$N_{j+1}$")
    plt.plot(t,E(xN,v_N[i]),label="E(t)")
    print(E(xN,v_N[i]))
    plt.title(f"Coupled Harmonic Oscillator for {len(new_x)} Particles with initial config 2 and j=N/2")
    plt.ylabel("x(t)")
    plt.xlabel("t")
    #plt.ylim(-2,2)
    plt.xlim(0,10)
    plt.grid()
    if len(new_x) == 4:
        plt.legend()
    plt.savefig(f"ex_33_{i+1}.png",bbox_inches="tight")
    plt.show()
# %%
