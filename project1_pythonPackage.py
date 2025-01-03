from dolfin import *
import numpy as np

# Define mesh size (more triangles => finer mesh)
mesh_resolution = 32  # You can change this value to adjust the number of triangles
mesh = UnitSquareMesh(mesh_resolution, mesh_resolution)

# Define function space
V = FunctionSpace(mesh, "P", 1)  # Linear Lagrange elements

# Define boundary condition
u_D = Expression("1 + x[0]*x[0] + 2*x[1]*x[1]", degree=2)  # Example Dirichlet BC
bc = DirichletBC(V, u_D, "on_boundary")

# Define source term
f = Constant(-6.0)  # Source term

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
a = dot(grad(u), grad(v)) * dx
L = f * v * dx

# Compute solution
u = Function(V)
solve(a == L, u, bc)

# Save solution to file
file = File("solution.pvd")
file << u

# Plot solution
import matplotlib.pyplot as plt
plot(u)
plt.title("FEM Solution")
plt.show()

# Optional: Number of triangles in the mesh
print("Number of triangles in the mesh:", mesh.num_cells())
