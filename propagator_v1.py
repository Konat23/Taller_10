import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from utils import plot_frame, timing

Tout = 1000
Nx = 200
Nz = 200
c = 1200.0
dh = 2.0
dt = dh / (c * np.sqrt(2))
alpha = 0.15
G = c * dt / dh

Ix0 = np.array([Nx // 2], dtype=np.int64)
Iz0 = Nz // 3

x0 = dh * (Ix0[0] + 1)
z0 = dh * (Iz0 + 1)

iz = np.arange(Nz)
jx = np.arange(Nx)
Z, X = np.meshgrid(iz, jx, indexing="ij")
r2 = (dh * (Z + 1) - z0) ** 2 + (dh * (X + 1) - x0) ** 2

fq = 20.0
t = np.arange(0, Tout * dt, dt)
src1d = (1 - 2 * (np.pi * fq * (t - 1.0 / fq)) ** 2) * np.exp(
    -((np.pi * fq * (t - 1.0 / fq)) ** 2)
)

src = src1d.reshape(-1, 1)

vp_ori = np.zeros((Nx, Nz), dtype=np.float64)
for iz_idx in range(Nz):
    vp_ori[:, iz_idx] = c


@jit(nopython=True)
def get_CPML(CPMLimit, R, Vmax, Nx, Nz, dx, dz, dt, f0):
    a_x = np.zeros(Nx, dtype=np.float64)
    a_x_half = np.zeros(Nx, dtype=np.float64)
    b_x = np.ones(Nx, dtype=np.float64)
    b_x_half = np.ones(Nx, dtype=np.float64)
    a_z = np.zeros(Nz, dtype=np.float64)
    a_z_half = np.zeros(Nz, dtype=np.float64)
    b_z = np.ones(Nz, dtype=np.float64)
    b_z_half = np.ones(Nz, dtype=np.float64)

    kappa_max = 1.0
    alpha_max = np.pi * f0 if f0 > 0.0 else 0.0
    d0_x = -3.0 * Vmax * np.log(R) / (2.0 * CPMLimit * dx)
    d0_z = -3.0 * Vmax * np.log(R) / (2.0 * CPMLimit * dz)

    for i in range(Nx):
        if i < CPMLimit:
            x = (CPMLimit - i) / CPMLimit
            sigma = d0_x * x * x
            b_x[i] = np.exp(-(sigma / kappa_max + alpha_max) * dt)
            a_x[i] = sigma * (b_x[i] - 1.0) / (sigma + kappa_max * alpha_max + 1e-20)
            b_x_half[i] = b_x[i]
            a_x_half[i] = a_x[i]
        elif i >= Nx - CPMLimit:
            idx_mirror = Nx - 1 - i
            if idx_mirror < CPMLimit:
                x = (CPMLimit - idx_mirror) / CPMLimit
                sigma = d0_x * x * x
                b_x[i] = np.exp(-(sigma / kappa_max + alpha_max) * dt)
                a_x[i] = (
                    sigma * (b_x[i] - 1.0) / (sigma + kappa_max * alpha_max + 1e-20)
                )
                b_x_half[i] = b_x[i]
                a_x_half[i] = a_x[i]

    for j in range(Nz):
        if j < CPMLimit:
            z = (CPMLimit - j) / CPMLimit
            sigma = d0_z * z * z
            b_z[j] = np.exp(-(sigma / kappa_max + alpha_max) * dt)
            a_z[j] = sigma * (b_z[j] - 1.0) / (sigma + kappa_max * alpha_max + 1e-20)
            b_z_half[j] = b_z[j]
            a_z_half[j] = a_z[j]
        elif j >= Nz - CPMLimit:
            idx_mirror = Nz - 1 - j
            if idx_mirror < CPMLimit:
                z = (CPMLimit - idx_mirror) / CPMLimit
                sigma = d0_z * z * z
                b_z[j] = np.exp(-(sigma / kappa_max + alpha_max) * dt)
                a_z[j] = (
                    sigma * (b_z[j] - 1.0) / (sigma + kappa_max * alpha_max + 1e-20)
                )
                b_z_half[j] = b_z[j]
                a_z_half[j] = a_z[j]

    return a_x, a_x_half, b_x, b_x_half, a_z, a_z_half, b_z, b_z_half


@timing
@jit(nopython=True)
def propagate_wave_cpml(Tout, Nx, Nz, c, dh, dt, G, src):
    P1 = np.zeros((Nz, Nx), dtype=np.float64)
    P2 = np.zeros((Nz, Nx), dtype=np.float64)
    P3 = np.zeros((Nz, Nx), dtype=np.float64)

    video = np.zeros((Tout, Nz, Nx), dtype=np.float32)
    video[0, :, :] = P2.astype(np.float32)

    coeff_center = 2.0 - 4.0 * G * G
    coeff_neigh = G * G

    CPMLimit = 20
    Vmax = c
    R = 1e-3
    a_x, a_x_half, b_x, b_x_half, a_z, a_z_half, b_z, b_z_half = get_CPML(
        CPMLimit, R, Vmax, Nx, Nz, dh, dh, dt, fq
    )

    for t_idx in range(1, Tout):
        P3_inner = (
            coeff_center * P2[1:-1, 1:-1]
            + coeff_neigh
            * (P2[2:, 1:-1] + P2[:-2, 1:-1] + P2[1:-1, 2:] + P2[1:-1, :-2])
            - P1[1:-1, 1:-1]
        )
        P3[1:-1, 1:-1] = P3_inner

        for ix in range(1, Nx - 1):
            bzh = b_z_half[Nz - 1]
            azh = a_z_half[Nz - 1]
            P3[ix, Nz - 1] += bzh * P3[ix, Nz - 2] + azh * P3[ix, Nz - 2]

        for iz in range(1, Nz - 1):
            bxl = b_x_half[0]
            axl = a_x_half[0]
            P3[0, iz] += bxl * P3[1, iz] + axl * P3[1, iz]

            bxr = b_x_half[Nx - 1]
            axr = a_x_half[Nx - 1]
            P3[Nx - 1, iz] += bxr * P3[Nx - 2, iz] + axr * P3[Nx - 2, iz]

        P1[:, :] = P2
        P2[:, :] = P3

        P2[Iz0, Ix0[0]] += src[t_idx, 0]
        video[t_idx, :, :] = P3.astype(np.float32)

        if (t_idx % 100) == 0:
            print(f"Iteration {t_idx}/{Tout}")

    return video


video = propagate_wave_cpml(Tout, Nx, Nz, c, dh, dt, G, src)

frame = 200
plot_frame(video, n_save=frame, title=f"Δt = {dt:.6e} s, frame = {frame}")
