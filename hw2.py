import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
def tridiag(num_main,num_up,num_down,dim):
    diagnoses=[np.full(dim,num_main),np.full(dim-1,num_up),np.full(dim-1,num_down)]
    sets=[0,1,-1]
    C=diags(diagnoses,sets,format='csr')
    return C

N_set=[20,40,80,160]
res=[]

for i in range(len(N_set)):
    u0 = lambda x, y: np.sin(x) * np.sin(y)
    u_f = lambda x, y, t: np.exp(-2 * t) * np.sin(x) * np.sin(y)
    T, N = 1, N_set[i]
    x, y = np.linspace(0, np.pi, N + 1), np.linspace(0, np.pi, N + 1)
    X, Y = np.meshgrid(x, y)
    u = u0(X, Y)
    u_t = np.zeros_like(u)

    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dt = 0.5 * dx ** 2  # time step size, thus r_x=r_y=0.5, r=0.5*rx
    r = 0.25
    steps = int(T / dt)

    A_plus = tridiag(1 - 2 * r, r, r, N + 1)  # I+D_xD_-x or I+DyD-y
    A_minus = tridiag(1 + 2 * r, -r, -r, N + 1)  # I-D_xD_-x or I-DyD-y

    for n in range(steps):
        for i in range(1, N):
            u_t[i, :] = spsolve(A_minus, A_plus @ u[i, :].T).T
        u_t[:, 0] = np.zeros_like(u[:, 0])
        u_t[:, N] = np.zeros_like(u[:, N])
        for j in range(1, N):
            u[:, j] = spsolve(A_minus, A_plus @ u_t[:, j])
        u[0, :] = np.zeros_like(u_t[0, :])
        u[N, :] = np.zeros_like(u_t[N, :])

    u_final = u_f(X, Y, 1)
    error=u_final-u
    res.append(np.max(np.abs(error)))
res=np.array(res)
plt.figure(1)
plt.contourf(X,Y,u)
plt.colorbar()
plt.figure(2)
plt.contourf(X,Y,u_final)
plt.colorbar()

plt.figure(3)
plt.plot(np.log(-np.log(res)))
plt.show()




