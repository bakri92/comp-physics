
import numpy as np
import matplotlib.pyplot as plt
#%%
"""
exercise 1: 

"""
#%%
## 1:
matrikel="421409"
np.random.seed(1409)
A = np.random.uniform(low=-5.0, high=5.0, size=(6,6))
print(A)
# %%
## 2:

max_val = np.max(A)
max_idx = np.where(A == max_val)
print(f"Index of the largest value in the matrix A is: {max_idx[0][0],max_idx[1][0]}, and its value is:{max_val}")
# %%
# 3:
larget_in_col_row = np.max(A,axis=0)
larget_in_row_col = np.max(A,axis=1)
print(larget_in_col_row)
print(larget_in_row_col)

print(f"Multiplying the column and the row as dot product is: {np.dot(larget_in_row_col,larget_in_col_row)}")
print(f"Multiplying the row and the column as outer product is: {np.outer(larget_in_col_row,larget_in_row_col)}")

#%%
# 4: 
B = np.random.uniform(low=-5.0, high=5.0, size=(6,6))
print(B)
C = np.dot(A,B)
D = np.dot(B,A)
print(A[1])
# %%
"""
exercise 2: 
"""
#%%
# 1: 

def cheby(x,N):
    ## define the recursive chebyshev function
    def chebyshev_poly(x, n):
        if n == 0:
            return 1
        elif n == 1:
            return x
        else:
            return 2 * x * chebyshev_poly(x, n - 1) - chebyshev_poly(x, n - 2)
    ## define a matrix
    Matrix = np.zeros((len(x),(N+1)))  
    ## fill the matrix with chebyshev polynomials for all x values 
    ## for all degrees of N and smaller to zero 
    for ns in range(N+1):
        for i, xs in enumerate(x):
            Matrix[i,ns] = chebyshev_poly(xs, ns)
    return Matrix

## create x of length 1000
X = np.linspace(-1,1,1000)
## define the orders of polynomials
N = 5
cheby_matrix = cheby(X,N)
# %%
## 2: 
T0 = cheby_matrix[:,0]
T1 = cheby_matrix[:,1]
T2 = cheby_matrix[:,2]
T3 = cheby_matrix[:,3]
T4 = cheby_matrix[:,4]

plt.plot(X,T0, label="$T_0(x)$")
plt.ylabel("$T(x)$")
plt.xlabel('X')

plt.plot(X,T1,label="$T_1(x)$")
plt.plot(X,T2,  label="$T_2(x)$")
plt.plot(X,T3,  label="$T_3(x)$")
plt.plot(X,T4,label="$T_4(x)$")
plt.xlim(-1.5,1)
plt.legend()
plt.title("Chebyshev Polynomial of Degrees 0 to 4")
plt.savefig("polynomials.pdf", bbox_inches="tight")
plt.show()