import numpy as np
from numpy.linalg import eigh
import matplotlib.pyplot as plt
from scipy.special import eval_hermite
from scipy import integrate
from math import factorial

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

def H4(N):
    matP2 = P2(N+4)
    matX = X(N+4)

    #ans = matP2 + matX.dot(matX).dot(matX).dot(matX)
    ans = matP2 + matX @ matX @ matX @ matX

    return ans[:N, :N]

def phi(n, x):
    norm = np.sqrt(2**n * factorial(n) * np.sqrt(np.pi))
    fun = np.exp(-x**2/2) * eval_hermite(n, x) / norm

    return  fun

if __name__ == '__main__':
    N = 200
    #val, vec = np.linalg.eigh(H4(N))

    #X = np.linspace(-5,5,1000)

    #vec0 = vec[:,0]

    npoints = 1000
    Xp = np.linspace(-5,5,npoints)

    # Verify that oscarm eigenfunctions are orthonormal
    for n in range(5):
        result = integrate.quad(lambda x: phi(n,x)**2,-np.inf,np.inf)
        print(n, result)

    # Plot the oscarm eigenfunctions
    #for n in range(5):
    #    Y = phi(n, Xp)

    #    plt.plot(Xp, Y, label = 'n = ' + str(n))

    #plt.legend()
    #plt.show()

    # Now for the computation of the quartic oscillator eigenfunctions
    N = 120
    val, vec = eigh(H4(N))

    n_wfns = 3
    C = vec[:,:n_wfns]

    phi_ij = np.zeros((N,npoints))

    for n in range(N):
        phi_ij[n,:] = phi(n, Xp)

    psi0 = C.T @ phi_ij

    for k in range(n_wfns):
        phi0 = phi(k, Xp)

        sign1 = np.sign(phi0[npoints//2])
        sign2 = np.sign(psi0[k,npoints//2])

        p = plt.plot(Xp, sign2*psi0[k, :], label = 'k = ' + str(k)) 
        color = p[0].get_color()
        plt.plot(Xp, sign1*phi0, '--', label = 'k = ' + str(k), color = color)

    plt.legend()
    plt.show()
