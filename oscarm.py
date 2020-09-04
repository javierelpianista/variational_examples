import numpy as np
from math import factorial
from numpy.linalg import eigvals
import matplotlib.pyplot as plt

# Matrix elements of the position operator
def x_elem(m,n):
    if m == n - 1:
        return np.sqrt(n/2.)
    elif m == n + 1:
        return np.sqrt((n+1)/2.)
    else:
        return 0

# Matrix elements of the momentum operator
def p_elem(m,n):
    if m == n - 1:
        return -1j*np.sqrt(n/2.)
    elif m == n + 1:
        return 1j*np.sqrt((n+1)/2.)
    else:
        return 0

# Matrix representation of the position operator
def X(N):
    mat = np.zeros([N, N])
    for m in range(N):
        for n in range(N):
            mat[m, n] = x_elem(m, n)

    return mat

# Matrix representation of the momentum operator
def P(N):
    mat = np.zeros([N, N], dtype=complex)
    for m in range(N):
        for n in range(N):
            mat[m, n] = p_elem(m, n)

    return mat

# Since the square of the momentum operator is real, we can discard the complex part
def P2(N):
    Pmat = P(N)
    #ans_complex = P(N).dot(P(N))
    ans_complex = Pmat.dot(Pmat)

    return np.array(ans_complex, dtype=float)

# This shows that the harmonic oscillator Hamiltonian is diagonal in the basis of its
# eigenvectors
print(P2(5) + X(5).dot(X(5)))

def H4(N):
    matP2 = P2(N+4)
    matX = X(N+4)

    ans = matP2 + matX.dot(matX).dot(matX).dot(matX)

    return ans[:N, :N]

ref = 1.06036209048418289964704601
varX, varY = [], []
for N in range(2, 51):
    vals = eigvals(H4(N))
    val = min(vals)
    err = -np.log10(np.abs(ref - val))
    varX.append(N)
    varY.append(err)
    print('{:2d}{:18.12f}{:8.1f}'.format(N, val, err))
