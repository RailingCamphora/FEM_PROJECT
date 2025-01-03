import numpy as np


def jacobi(u, b, N):
    """Performs two Jacobi iterations."""
    u_new = u.copy()
    for _ in range(2):  # Two Jacobi iterations
        for i in range(1, N):
            u_new[i] = 0.5 * (u[i - 1] + u[i + 1] - b[i])
        u = u_new.copy()
    return u


def compute_residual(u, b, N):
    """Compute the residual of the current solution."""
    r = np.zeros(N + 1)
    for i in range(1, N):
        r[i] = u[i - 1] - 2 * u[i] + u[i + 1] - b[i]
    return r


def multigrid(N):
    dx = 1.0 / N
    x = np.linspace(0, 1, N + 1)
    u = np.zeros(N + 1)
    b = np.ones(N + 1) * dx ** 2

    iterations = 0
    while True:
        u = jacobi(u, b, N)
        r = compute_residual(u, b, N)
        res_norm = np.linalg.norm(r[1:N]) / np.linalg.norm(b[1:N])
        if res_norm < 1e-8:
            break
        iterations += 1

    return u, iterations


# Testing the multigrid solver
for N in [50, 100, 200]:
    u, iters = multigrid(N)
    print(f"N={N}, Iterations={iters}")
