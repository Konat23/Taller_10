import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from utils import plot_frame, timing
from utils import get_CPML

# --- parámetros ---
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
a = (np.pi * fq) ** 2
t0 = 0.1
src1d = -2 * a * (t - t0) * np.exp(-a * (t - t0) ** 2)

src = src1d.reshape(-1, 1)

vp_ori = np.zeros((Nx, Nz), dtype=np.float64)
for iz_idx in range(Nz):
    vp_ori[:, iz_idx] = c


# @timing
# @jit(nopython=True)
def propagate_wave_cpml(Tout, Nx, Nz, c, dh, dt, G, src):
    """
    Versión corregida: usa convención (ix, iz) -> arrays shape (Nx, Nz)
    para mantener coherencia con get_CPML y con la implementación de CPML.
    """
    P1 = np.zeros((Nx, Nz), dtype=np.float64)
    P2 = np.zeros((Nx, Nz), dtype=np.float64)
    P3 = np.zeros((Nx, Nz), dtype=np.float64)

    video = np.zeros((Tout, Nz, Nx), dtype=np.float32)
    video[0, :, :] = P2.T.astype(np.float32)

    coeff_center = 2.0 - 4.0 * G * G
    coeff_neigh = G * G

    CPMLimit = 20
    Vmax = c
    R = 1e-3
    fq = 25.0
    Ix0 = Nx // 2
    Iz0 = Nz // 2

    a_x, a_x_half, b_x, b_x_half, a_z, a_z_half, b_z, b_z_half = get_CPML(
        CPMLimit, R, Vmax, Nx, Nz, dh, dh, dt, fq
    )

    F_dPdx = np.zeros((Nx, Nz), dtype=np.float64)
    F_dPdz = np.zeros((Nx, Nz), dtype=np.float64)
    F_d2Pdx2 = np.zeros((Nx, Nz), dtype=np.float64)
    F_d2Pdz2 = np.zeros((Nx, Nz), dtype=np.float64)

    one_over_dh = 1.0 / dh

    for t_idx in range(1, Tout):
        for ix in range(1, Nx - 1):
            for iz in range(1, Nz - 1):
                P3[ix, iz] = (
                    coeff_center * P2[ix, iz]
                    + coeff_neigh
                    * (
                        P2[ix + 1, iz]
                        + P2[ix - 1, iz]
                        + P2[ix, iz + 1]
                        + P2[ix, iz - 1]
                    )
                    - P1[ix, iz]
                )

        # --- CPML en el lado izquierdo (small ix) ---
        for ix in range(1, CPMLimit + 1):
            for iz in range(1, Nz - 1):
                dP_dx = (P2[ix + 1, iz] - P2[ix, iz]) * one_over_dh
                dP_dz = (P2[ix, iz + 1] - P2[ix, iz]) * one_over_dh

                F_dPdx[ix, iz] = F_dPdx[ix, iz] * b_x_half[ix] + a_x_half[ix] * dP_dx
                dP_dx_mod = dP_dx + F_dPdx[ix, iz]

                F_dPdz[ix, iz] = F_dPdz[ix, iz] * b_z[iz] + a_z[iz] * dP_dz
                dP_dz_mod = dP_dz + F_dPdz[ix, iz]

                d2P_dx2 = (
                    dP_dx_mod - (P2[ix, iz] - P2[ix - 1, iz]) * one_over_dh
                ) * one_over_dh
                d2P_dz2 = (
                    dP_dz_mod - (P2[ix, iz] - P2[ix, iz - 1]) * one_over_dh
                ) * one_over_dh

                F_d2Pdx2[ix, iz] = F_d2Pdx2[ix, iz] * b_x[ix] + a_x[ix] * d2P_dx2
                d2P_dx2_mod = d2P_dx2 + F_d2Pdx2[ix, iz]

                F_d2Pdz2[ix, iz] = F_d2Pdz2[ix, iz] * b_z[iz] + a_z[iz] * d2P_dz2
                d2P_dz2_mod = d2P_dz2 + F_d2Pdz2[ix, iz]

                laplacian_mod = d2P_dx2_mod + d2P_dz2_mod
                P3[ix, iz] = (
                    c * c * laplacian_mod * dt * dt + 2 * P2[ix, iz] - P1[ix, iz]
                )

        # --- CPML en el lado derecho ---
        for ix in range(Nx - 2, Nx - CPMLimit - 2, -1):
            for iz in range(1, Nz - 1):
                dP_dx = (P2[ix, iz] - P2[ix - 1, iz]) * one_over_dh
                dP_dz = (P2[ix, iz + 1] - P2[ix, iz]) * one_over_dh

                F_dPdx[ix, iz] = (
                    F_dPdx[ix, iz] * b_x_half[ix - 1] + a_x_half[ix - 1] * dP_dx
                )
                dP_dx_mod = dP_dx + F_dPdx[ix, iz]

                F_dPdz[ix, iz] = F_dPdz[ix, iz] * b_z[iz] + a_z[iz] * dP_dz
                dP_dz_mod = dP_dz + F_dPdz[ix, iz]

                d2P_dx2 = (
                    (P2[ix + 1, iz] - P2[ix, iz]) * one_over_dh - dP_dx_mod
                ) * one_over_dh
                d2P_dz2 = (
                    dP_dz_mod - (P2[ix, iz] - P2[ix, iz - 1]) * one_over_dh
                ) * one_over_dh

                F_d2Pdx2[ix, iz] = F_d2Pdx2[ix, iz] * b_x[ix] + a_x[ix] * d2P_dx2
                d2P_dx2_mod = d2P_dx2 + F_d2Pdx2[ix, iz]

                F_d2Pdz2[ix, iz] = F_d2Pdz2[ix, iz] * b_z[iz] + a_z[iz] * d2P_dz2
                d2P_dz2_mod = d2P_dz2 + F_d2Pdz2[ix, iz]

                laplacian_mod = d2P_dx2_mod + d2P_dz2_mod
                P3[ix, iz] = (
                    c * c * laplacian_mod * dt * dt + 2 * P2[ix, iz] - P1[ix, iz]
                )

        # --- CPML en la parte inferior (bordes z grandes) ---
        for ix in range(CPMLimit + 1, Nx - CPMLimit - 1):
            for iz in range(Nz - CPMLimit - 1, Nz - 1):
                dP_dx = (P2[ix + 1, iz] - P2[ix, iz]) * one_over_dh
                dP_dz = (P2[ix, iz] - P2[ix, iz - 1]) * one_over_dh

                F_dPdx[ix, iz] = F_dPdx[ix, iz] * b_x[ix] + a_x[ix] * dP_dx
                dP_dx_mod = dP_dx + F_dPdx[ix, iz]

                F_dPdz[ix, iz] = (
                    F_dPdz[ix, iz] * b_z_half[iz - 1] + a_z_half[iz - 1] * dP_dz
                )
                dP_dz_mod = dP_dz + F_dPdz[ix, iz]

                d2P_dx2 = (
                    dP_dx_mod - (P2[ix, iz] - P2[ix - 1, iz]) * one_over_dh
                ) * one_over_dh
                d2P_dz2 = (
                    (P2[ix, iz + 1] - P2[ix, iz]) * one_over_dh - dP_dz_mod
                ) * one_over_dh

                F_d2Pdx2[ix, iz] = F_d2Pdx2[ix, iz] * b_x[ix] + a_x[ix] * d2P_dx2
                d2P_dx2_mod = d2P_dx2 + F_d2Pdx2[ix, iz]

                F_d2Pdz2[ix, iz] = F_d2Pdz2[ix, iz] * b_z[iz] + a_z[iz] * d2P_dz2
                d2P_dz2_mod = d2P_dz2 + F_d2Pdz2[ix, iz]

                laplacian_mod = d2P_dx2_mod + d2P_dz2_mod
                P3[ix, iz] = (
                    c * c * laplacian_mod * dt * dt + 2 * P2[ix, iz] - P1[ix, iz]
                )

        # avanzar en el tiempo
        P1[:, :] = P2
        P2[:, :] = P3

        # inyectar fuente (ojo: ahora P2[ix, iz] convención)
        P2[Ix0, Iz0] += src[t_idx, 0]

        # guardar frame transpuesto para mantener (Nz, Nx) en video
        video[t_idx, :, :] = P3.T.astype(np.float32)

        if (t_idx % 100) == 0:
            print(f"Iteration {t_idx}/{Tout}")

    return video


# ejecutar
video = propagate_wave_cpml(Tout, Nx, Nz, c, dh, dt, G, src)

# graficar algunos frames (mantengo tu plot_frame)
frames = [100, 150, 200, 250, 300]
vmax = 1
for frame in frames:
    plot_frame(
        video, n_save=frame, title=f"Δt = {dt:.6e} s, frame = {frame}", vmax=vmax
    )
