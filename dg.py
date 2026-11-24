import numpy as np
import matplotlib.pyplot as plt

N = 100
p = 1         
xmin = 0.0
xmax = 2.0 * np.pi
CFL = 0.15
T = 0.5
h = (xmax - xmin) / N
xc = xmin + (np.arange(N) + 0.5) * h # Element centers
print("xc none", xc[:, None])
print("xc", xc)

# IC
def u0(x):
    return np.sin(x)

# p=1 Legendre basis: phi0 = 1, phi1 = xi
# Gauss quadrature on reference element [-1,1]

xi_q, w_q = np.polynomial.legendre.leggauss(3)
print("xi_q", xi_q[None, :]*h) #row vector
phi0 = np.ones_like(xi_q)
phi1 = xi_q.copy()
xq = xc[:, None] + 0.5 * h * xi_q[None, :] # Physical coordinates at quadrature points

# L2 projection of initial condition onto P1 DG basis
u_init = u0(xq) #sin wave at quadrature pts
a = (0.5 * np.sum(w_q[None, :] * u_init, axis=1))
print("a", a)
b = (1.5 * np.sum( w_q[None, :] * u_init * xi_q[None, :],axis=1))

#Burgers flux
def flux(u):
    return 0.5 * u**2

#local LF numerical flux
def numerical_flux(uL, uR):
    alpha = np.maximum(np.abs(uL), np.abs(uR))
    return 0.5 * (flux(uL) + flux(uR)) - 0.5 * alpha * (uR - uL)


def rhs(a, b):
    u_right = a+b # right boundary bc reference element = 1
    print("u_right", u_right)
    u_left  = a-b # left boundary bc reference element = -1

    # F[i] = flux at interface between element i and i+1
    # Periodic BCs with np.roll
    F = numerical_flux(u_right, np.roll(u_left, -1))
    F_left = np.roll(F, 1)
    F_right = F
    # volume integral
    uq = a[:, None] + b[:, None] * xi_q[None, :]
    fq = flux(uq)
    volume = np.sum(w_q[None, :] * fq, axis=1)
    # residual
    rhs_a = F_left - F_right
    rhs_b = volume - F_left - F_right
    #inverse mass matrix to isolate: M du/dt = RHS, M = diag(h, h/3)
    da_dt = rhs_a / h
    db_dt = 3.0 * rhs_b / h
    return da_dt, db_dt

#SSPRK3
def ssprk3(a, b, dt):
    # Stage 1
    da, db = rhs(a, b)
    a1 = a + dt * da
    b1 = b + dt * db

    # Stage 2
    da, db = rhs(a1, b1)
    a2 = 0.75 * a + 0.25 * (a1 + dt * da)
    b2 = 0.75 * b + 0.25 * (b1 + dt * db)

    # Stage 3
    da, db = rhs(a2, b2)
    a_new = (1.0 / 3.0) * a + (2.0 / 3.0) * (a2 + dt * da)
    b_new = (1.0 / 3.0) * b + (2.0 / 3.0) * (b2 + dt * db)
    return a_new, b_new

# go throuigh times
t = 0.0
snapshots = {}
save_times = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
snapshots[0.0] = (a.copy(), b.copy())

while t < T:
    # max Burgers wave speed
    umax = np.max(np.abs(a + b))
    umax = max(umax, np.max(np.abs(a - b)), 1e-12)
    # CFL condition for P1 DG
    dt = CFL * h/(3.0 * umax)
    dt = min(dt, T-t) # dont step past final time
    a, b = ssprk3(a, b, dt)
    t += dt

    for ts in save_times:
        if ts not in snapshots and t >= ts:
            snapshots[ts] = (a.copy(), b.copy())

print(f"Final time: {t:.4f}")
print(f"Number of elements: {N}")
print(f"Polynomial degree: {p}")
print(f"Final maximum solution: {np.max(np.abs(a)):.4f}")


xi_plot = np.linspace(-1, 1, 20)
fig, axes = plt.subplots(3, 2, figsize=(14, 9))
axes = axes.flatten()

for ax, ts in zip(axes, save_times):
    a_snap, b_snap = snapshots[ts]
    x_plot = xc[:, None] + 0.5 * h * xi_plot[None, :]
    u_plot = a_snap[:, None] + b_snap[:, None] * xi_plot[None, :]
    for i in range(N):
        ax.plot(x_plot[i], u_plot[i], color="blue")
    # Initial condition
    x_exact = np.linspace(xmin, xmax, 500)
    ax.plot(x_exact, u0(x_exact), "k--", alpha=0.5)
    ax.set_title(f"t = {ts:.2f}")
    ax.set_xlabel("x")
    ax.set_ylabel("u(x,t)")
    ax.grid(True)

plt.tight_layout()
plt.show()
