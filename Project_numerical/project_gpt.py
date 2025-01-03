import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import matplotlib.tri as tri




## the f correspond to u=xy(1-x)(1-y)exp(-(x-0.5)^2-(y-0.5)^2) is

def create_uniform_mesh(n):
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, y)
    X = X.flatten()
    Y = Y.flatten()
    triangles = []
    for i in range(n - 1):
        for j in range(n - 1):
            idx = i * n + j
            triangles.append([idx, idx + 1, idx + n])
            triangles.append([idx + 1, idx + n + 1, idx + n])
    return X, Y, np.array(triangles)

def local_stiffness_matrix(x, y):
    # Coordinates of the triangle nodes
    C = np.array([[1, x[0], y[0]],
                  [1, x[1], y[1]],
                  [1, x[2], y[2]]])
    # Area of the triangle
    area = 0.5 * np.abs(np.linalg.det(C))
    # Gradients
    invC = np.linalg.inv(C)
    grad = invC[1:, :]
    # Local stiffness matrix
    K = area * (grad @ grad.T)
    return K

def assemble_global_matrix(X, Y, triangles):
    N = len(X)
    K = lil_matrix((N, N))
    for tri in triangles:
        vert_indices = tri
        verts = np.vstack((X[vert_indices], Y[vert_indices])).T
        K_local = local_stiffness_matrix(verts[:, 0], verts[:, 1])
        for i in range(3):
            for j in range(3):
                K[tri[i], tri[j]] += K_local[i, j]
    return K

def apply_boundary_conditions(K, f, X, Y):
    boundary_indices = np.where((X == 0) | (X == 1) | (Y == 0) | (Y == 1))[0]
    for idx in boundary_indices:
        K[idx, :] = 0
        K[:, idx] = 0
        K[idx, idx] = 1
        f[idx] = 0
    return K, f

n = 20  # Number of points in each dimension
X, Y, triangles = create_uniform_mesh(n)
K = assemble_global_matrix(X, Y, triangles)
f = np.ones(X.shape[0])  # Load vector

K, f = apply_boundary_conditions(K, f, X, Y)

# Solve the linear system
u = spsolve(K.tocsr(), f)

# Plot the solution
fig, ax = plt.subplots()
tpc = tri.Triangulation(X, Y, triangles)
contour = ax.tricontourf(tpc, u, levels=40, cmap="viridis")
fig.colorbar(contour, ax=ax, label="Solution u")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title("FEM Solution of the Poisson Equation")
plt.show()
