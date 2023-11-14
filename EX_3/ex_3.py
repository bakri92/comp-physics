#%%
import numpy as np
import matplotlib.pyplot as plt

def Analyzer(c, s, cHWP, sHWP, T0):
    # EOM: plane rotation
    c2 = cHWP*c + sHWP*s
    s2 = -sHWP*c + cHWP*s
    x = c2*c2 - s2*s2 # cos(2(x-a))
    y = 2*c2*s2 # sin(2(x-a))
    
    # Malus law
    r0 = np.random.rand()
    if x > 2*r0 - 1:
        j = 0 # +1 event
    else:
        j = -1 # -1 event
    
    # Time delay
    r1 = np.random.rand()
    l = y*y*y*y*T0*r1 # delay time: T_0 sin(2*(theta1 - x))**4
    
    return j, l

np.random.seed(6653)
Nsamples = 100000
nsteps = 32
T0 = 1000  # ns, maximum time delay
W = 1
pi = np.arccos(-1)
twopi = 2 * pi
HWP2 = 0
cHWP2 = np.cos(HWP2 * pi / 180)
sHWP2 = np.sin(HWP2 * pi / 180)
count = np.zeros((2, 2, 2, nsteps), dtype=int)
E1 = np.zeros((2, nsteps))
E2 = np.zeros((2, nsteps))
E12 = np.zeros((4, nsteps))
tot = np.zeros((2, nsteps), dtype=int)

# first step: simulate the event
for ipsi0 in range(nsteps):
    cHWP1 = np.cos(ipsi0 * twopi / nsteps)
    sHWP1 = np.sin(ipsi0 * twopi / nsteps)
    for i in range(Nsamples):
        r0 = np.random.rand()
        c1 = np.cos(r0 * twopi)  # polarization angle x of particle going to station 1
        s1 = np.sin(r0 * twopi)
        c2 = -s1  # polarization angle x + pi/2 of particle going to station 2
        s2 = c1
        # first station
        j1, l1 = Analyzer(c1, s1, cHWP1, sHWP1, T0)
        # second station
        j2, l2 = Analyzer(c2, s2, cHWP2, sHWP2, T0)
        count[j1, j2, 0, ipsi0] += 1  # Malus law model
        if abs(l1 - l2) < W:
            count[j1, j2, 1, ipsi0] += 1
        


# second step: data analysis
for j in range(nsteps):
    for i in range(2):
        tot[i, j] = np.sum(count[:, :, i, j])
        E12[i, j] = (
            count[0, 0, i, j]
            + count[1, 1, i, j]
            - count[1, 0, i, j]
            - count[0, 1, i, j]
        )
        E1[i, j] = (
            count[0, 0, i, j]
            + count[0, 1, i, j]
            - count[1, 1, i, j]
            - count[1, 0, i, j]
        )
        E2[i, j] = (
            count[0, 0, i, j]
            + count[1, 0, i, j]
            - count[1, 1, i, j]
            - count[0, 1, i, j]
        )
        if tot[i, j] > 0:
            E12[i, j] /= tot[i, j]
            E1[i, j] /= tot[i, j]
            E2[i, j] /= tot[i, j]
#%%
r0 = -np.cos(2*np.arange(33)*twopi/nsteps)

# %%
plt.plot(np.linspace(0,360,32),E12[0,:],"o",label="coincidence counting")
plt.plot(np.linspace(0,360,32),E12[0,:],"--")

plt.plot(np.linspace(0,360,32),E12[1,:],"o",label="no coincidence counting")
plt.plot(np.linspace(0,360,32),E12[1,:],"--")

plt.grid()
plt.legend()
plt.xlabel(r"$\varphi$ (degree)")
plt.ylabel(r"$<S_1.a_1 S_2.a_2>$")
plt.title(r"$E_{12}(a,b)$")
plt.savefig("E12.png",bbox_inches="tight")

plt.show()
#plt.plot(np.linspace(0,360,33),r0)
#%%
plt.plot(np.linspace(0,360,32),E1[0,:]*E2[0,:],"o",label="coincidence counting")
#plt.plot(np.linspace(0,360,32),E1[1,:]*E2[1,:],label="non coincidence counting")
plt.ylim(-1,1)
plt.grid()
plt.legend()
plt.xlabel(r"$\varphi$ (degree)")
plt.ylabel(r"$<S_1.a_1><S_2.a_2>$")
plt.title(r"$E_{1}(a,b)E_{2}(a,b)$")
plt.savefig("E1E2.png",bbox_inches="tight")
plt.show()
#plt.plot(np.linspace(0,360,33),r0)
# %%
