"""
Exercise 6: 
Yee Algorithm for simulating Maxwell equation 

"""
#%%
import numpy as np
import matplotlib.pyplot as plt 

#%% 

L = 5000 
delta = 0.02 
tau1 = 0.9 * delta
tau2 = 1.05 * delta
freq = 2 * np.pi
m = 10000 ## time steps 
jsource = 1000


Ex = np.zeros(L)
Hz = np.zeros(L)
Ex_prev = np.zeros(L)
Hz_prev = np.zeros(L)

def Source_Function(t:float):
	"""Wave Packet assuming f=1

	Parameters
	----------
	t : float
		time

	Returns
	-------
	float
		source function
	"""
	return np.sin(2*np.pi*t) * np.exp(-((t-30)/10)**2)


def sigma(x):
	if x >=0 and x<=6: 
		return 1 
	elif x>6 and x<L*delta-6: 
		return 0 
	else: 
		return 1 

def epsilon(x):
	if x >= 0 and x < L*delta/2:
		return 1
	elif L*delta/2 <= x and x < (L*delta/2)+2:
		return 1.46 ** 2
	else: 
		return 1 

def epsilon_thicc(x):
	if x >= 0 and x < L*delta/2:
		return 1
	elif L*delta/2 <= x and x < (L*delta/2)+2:
		return 1.46 ** 2
	else: 
		return 1 

def A(sigma_array, tau):
	nominator = 1 - (sigma_array * tau)/(2)
	denominator = 1 + (sigma_array * tau)/(2)
	return nominator / denominator

def B(sigma_array, tau):
	nominator = tau
	denominator = 1 + (sigma_array*tau)/(2)
	return nominator / denominator

def C(sigma_array, epsilon_array, tau):
	nominator = 1 - (sigma_array * tau)/(2*epsilon_array)
	denominator = 1 + (sigma_array * tau)/(2*epsilon_array)
	return nominator / denominator

def D(sigma_array, epsilon_array, tau):
	nominator = tau/epsilon_array
	denominator = 1 + (sigma_array * tau)/(2*epsilon_array)
	return nominator / denominator
	
## defining the simulation 
x = np.linspace(0,100,5000)
sigma_array = np.zeros(L)
for i in range(len(sigma_array)):
	sigma_array[i] = sigma(x[i])

epsilon_array = np.zeros(L)
for i in range(len(epsilon_array)):
	epsilon_array[i] = epsilon(x[i])

taus = tau1

A_array = A(sigma_array,tau1)
B_array = B(sigma_array, tau1)
C_array = C(sigma_array, epsilon_array, taus)
D_array = D(sigma_array, epsilon_array, taus)


#%%
## In the grid n is T in full time step 
## And j is halp space steps (l in the script)
for n in range(m):
	#Update magnetic field boundaries 
	Hz[L-1] = Hz_prev[L-2]
	#Update magnetic field 
	#for j in range(L-1):
	Hz[0:L-1] = A_array[0:L-1] * Hz_prev[0:L-1] + B_array[0:L-1] * (Ex[1:L] - Ex[0:L-1])/delta
	Hz_prev = Hz
	#Magnetic field source 
	Hz[jsource-1] -= Source_Function(n)
	Hz_prev[jsource-1] = Hz[jsource-1]
	#Update electric field boundaries 
	Ex[0] = Ex_prev[1]
	#Update electric field 
	#for j in range(1,L):
	Ex[1:L] = C_array[1:]  * Ex_prev[1:] + D_array[1:L] *( (Hz[1:L]-Hz[0:L-1])/delta -  Source_Function(n))
	Ex_prev = Ex
	Ex[jsource] += D(sigma_array[jsource],epsilon_array[jsource] , taus) * Source_Function((n)*taus)
	Ex_prev[jsource] = Ex[jsource]

	if  n == 0 or n == 1000 or n == 2000 or n == 5000: 
	#	plt.figure(figsize=(20,10))
		plt.plot(Ex,label="Wave")
		plt.grid()
		plt.title(fr"Transmision and reflection of light by a glass plate at t={n}, for $ \tau = 1.05 \Delta$")
		plt.plot(sigma_array-0.5, label="boundaries",alpha=0.5)
		plt.plot(epsilon_array-2, label="material boundaries",alpha=0.5)
		plt.ylim(-0.015,0.015)
		plt.legend()
		plt.xlabel("x")
		plt.ylabel("E")
		plt.savefig(f"Ex_tau105_t{n}.png")
		plt.show()
		plt.close()

# %%
epsilon_array_thicc = np.zeros(L)

def epsilon_thicc(x):
	if x >= 0 and x < L*delta/2:
		return 1
	elif L*delta/2 <= x :
		return 1.46 ** 2
	else: 
		return 1 


for i in range(len(epsilon_array_thicc)):
	epsilon_array_thicc[i] = epsilon_thicc(x[i])
#%%
taus = tau1

A_array = A(sigma_array,tau1)
B_array = B(sigma_array, tau1)
C_array = C(sigma_array, epsilon_array_thicc, taus)
D_array = D(sigma_array, epsilon_array_thicc, taus)

#%%
for n in range(m):
	#Update magnetic field boundaries 
	Hz[L-1] = Hz_prev[L-2]
	#Update magnetic field 
	#for j in range(L-1):
	Hz[0:L-1] = A_array[0:L-1] * Hz_prev[0:L-1] + B_array[0:L-1] * (Ex[1:L] - Ex[0:L-1])/delta
	Hz_prev = Hz
	#Magnetic field source 
	Hz[jsource-1] -= Source_Function(n)
	Hz_prev[jsource-1] = Hz[jsource-1]
	#Update electric field boundaries 
	Ex[0] = Ex_prev[1]
	#Update electric field 
	#for j in range(1,L):
	Ex[1:L] = C_array[1:]  * Ex_prev[1:] + D_array[1:L] *( (Hz[1:L]-Hz[0:L-1])/delta -  Source_Function(n))
	Ex_prev = Ex
	Ex[jsource] += D(sigma_array[jsource],epsilon_array[jsource] , taus) * Source_Function((n)*taus)
	Ex_prev[jsource] = Ex[jsource]

	if  n == 0 or n == 100 or n == 1000 or n == 2500 or n == 5000: 
	#	plt.figure(figsize=(20,10))
		plt.plot(Ex,label="Wave")
		plt.grid()
		plt.title(fr"Transmision and reflection of light by a glass plate at t={n}, for $ \tau = 1.05 \Delta$")
		plt.plot(sigma_array-0.5, label="boundaries",alpha=0.5)
		plt.plot(epsilon_array_thicc-2, label="material boundaries",alpha=0.5)
		plt.ylim(-0.015,0.015)
		plt.fill_between(np.arange(2500,5000), -0.015, 0.015,alpha=0.3, color="green")
		plt.legend()
		plt.xlabel("x")
		plt.ylabel("E")
	#	plt.savefig(f"Ex_tau09_t{n}_.png")
		plt.show()
		plt.close()
# %%
