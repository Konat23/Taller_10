import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from utils import plot_frame, timing

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
src1d = (1 - 2 * (np.pi * fq * (t - 1.0 / fq)) ** 2) * np.exp(
    -((np.pi * fq * (t - 1.0 / fq)) ** 2)
)

src = src1d.reshape(-1, 1)

vp_ori = np.zeros((Nx, Nz), dtype=np.float64)
for iz_idx in range(Nz):
    vp_ori[:, iz_idx] = c


@jit(nopython=True)
def get_CPML(CPMLimit, R, Vcpml, Nx, Nz, dx, dz, dt, frec):
    """
    Calcula coeficientes para CPML coherente con implementación MATLAB
    (idéntico a tu función original)
    """
    a_x = np.zeros(Nx, dtype=np.float64)
    a_x_half = np.zeros(Nx, dtype=np.float64)
    b_x = np.ones(Nx, dtype=np.float64)
    b_x_half = np.ones(Nx, dtype=np.float64)

    a_z = np.zeros(Nz, dtype=np.float64)
    a_z_half = np.zeros(Nz, dtype=np.float64)
    b_z = np.ones(Nz, dtype=np.float64)
    b_z_half = np.ones(Nz, dtype=np.float64)

    D_pml_x = CPMLimit * dx
    D_pml_z = CPMLimit * dz
    d0_x = -3.0 / (2.0 * D_pml_x) * np.log(R)
    d0_z = -3.0 / (2.0 * D_pml_z) * np.log(R)

    x = np.zeros(CPMLimit + 1)
    x_half = np.zeros(CPMLimit + 1)
    alpha_x = np.zeros(CPMLimit + 1)
    alpha_x_half = np.zeros(CPMLimit + 1)

    z = np.zeros(CPMLimit + 1)
    z_half = np.zeros(CPMLimit + 1)
    alpha_z = np.zeros(CPMLimit + 1)
    alpha_z_half = np.zeros(CPMLimit + 1)

    for j in range(CPMLimit + 1):
        x[j] = (CPMLimit - j) * dx
        z[j] = (CPMLimit - j) * dz
        alpha_x[j] = np.pi * frec * (D_pml_x - x[j]) / D_pml_x
        alpha_z[j] = np.pi * frec * (D_pml_z - z[j]) / D_pml_z

        x_half[j] = (CPMLimit - j) * dx - dx / 2.0
        z_half[j] = (CPMLimit - j) * dz - dz / 2.0
        alpha_x_half[j] = np.pi * frec * (D_pml_x - x_half[j]) / D_pml_x
        alpha_z_half[j] = np.pi * frec * (D_pml_z - z_half[j]) / D_pml_z

    # Bottom (bordes en z)
    for j in range(Nz - CPMLimit - 1, Nz):
        idx = Nz - j - 1
        if idx < len(z):
            d_z_val = d0_z * Vcpml * ((z[idx] / D_pml_z) ** 2)
            b_z[j] = np.exp(-(d_z_val + alpha_z[idx]) * dt)
            if np.abs(d_z_val + alpha_z[idx]) > 1e-20:
                a_z[j] = d_z_val / (d_z_val + alpha_z[idx]) * (b_z[j] - 1.0)
            else:
                a_z[j] = 0.0

            if j == Nz - CPMLimit - 1:
                b_z_half[j] = 0.0
                a_z_half[j] = 0.0
            else:
                d_z_half_val = d0_z * Vcpml * ((z_half[idx] / D_pml_z) ** 2)
                b_z_half[j] = np.exp(-(d_z_half_val + alpha_z_half[idx]) * dt)
                if np.abs(d_z_half_val + alpha_z_half[idx]) > 1e-20:
                    a_z_half[j] = (
                        d_z_half_val
                        / (d_z_half_val + alpha_z_half[idx])
                        * (b_z_half[j] - 1.0)
                    )
                else:
                    a_z_half[j] = 0.0

    # Left
    for i in range(CPMLimit + 1):
        if i < len(x):
            d_x_val = d0_x * Vcpml * ((x[i] / D_pml_x) ** 2)
            b_x[i] = np.exp(-(d_x_val + alpha_x[i]) * dt)
            if np.abs(d_x_val + alpha_x[i]) > 1e-20:
                a_x[i] = d_x_val / (d_x_val + alpha_x[i]) * (b_x[i] - 1.0)
            else:
                a_x[i] = 0.0

            if i == CPMLimit:
                b_x_half[i] = 0.0
                a_x_half[i] = 0.0
            else:
                d_x_half_val = d0_x * Vcpml * ((x_half[i] / D_pml_x) ** 2)
                b_x_half[i] = np.exp(-(d_x_half_val + alpha_x_half[i]) * dt)
                if np.abs(d_x_half_val + alpha_x_half[i]) > 1e-20:
                    a_x_half[i] = (
                        d_x_half_val
                        / (d_x_half_val + alpha_x_half[i])
                        * (b_x_half[i] - 1.0)
                    )
                else:
                    a_x_half[i] = 0.0

    # Right
    for i in range(Nx - CPMLimit - 1, Nx):
        idx = Nx - i - 1
        if idx < len(x):
            d_x_val = d0_x * Vcpml * ((x[idx] / D_pml_x) ** 2)
            b_x[i] = np.exp(-(d_x_val + alpha_x[idx]) * dt)
            if np.abs(d_x_val + alpha_x[idx]) > 1e-20:
                a_x[i] = d_x_val / (d_x_val + alpha_x[idx]) * (b_x[i] - 1.0)
            else:
                a_x[i] = 0.0

            if i == Nx - CPMLimit - 1:
                b_x_half[i] = 0.0
                a_x_half[i] = 0.0
            else:
                d_x_half_val = d0_x * Vcpml * ((x_half[idx] / D_pml_x) ** 2)
                b_x_half[i] = np.exp(-(d_x_half_val + alpha_x_half[idx]) * dt)
                if np.abs(d_x_half_val + alpha_x_half[idx]) > 1e-20:
                    a_x_half[i] = (
                        d_x_half_val
                        / (d_x_half_val + alpha_x_half[idx])
                        * (b_x_half[i] - 1.0)
                    )
                else:
                    a_x_half[i] = 0.0

    return a_x, a_x_half, b_x, b_x_half, a_z, a_z_half, b_z, b_z_half


@timing
@jit(nopython=True)
def propagate_wave_cpml(Tout, Nx, Nz, c, dh, dt, G, src):
    """
    Versión corregida: usa convención (ix, iz) -> arrays shape (Nx, Nz)
    para mantener coherencia con get_CPML y con la implementación de CPML.
    """
    # NOTE: cambio principal aquí: P arrays con forma (Nx, Nz)
    P1 = np.zeros((Nx, Nz), dtype=np.float64)
    P2 = np.zeros((Nx, Nz), dtype=np.float64)
    P3 = np.zeros((Nx, Nz), dtype=np.float64)

    # video lo guardamos al final con shape (Tout, Nz, Nx) para compatibilidad con plot_frame
    video = np.zeros((Tout, Nz, Nx), dtype=np.float32)
    video[0, :, :] = P2.T.astype(np.float32)

    coeff_center = 2.0 - 4.0 * G * G
    coeff_neigh = G * G

    CPMLimit = 20
    Vmax = c
    R = 1e-3
    fq = 25.0
    # Fuente en el centro (coordenadas en términos de indices ix, iz)
    Ix0 = Nx // 2
    Iz0 = Nz // 2

    a_x, a_x_half, b_x, b_x_half, a_z, a_z_half, b_z, b_z_half = get_CPML(
        CPMLimit, R, Vmax, Nx, Nz, dh, dh, dt, fq
    )

    # campos auxiliares para CPML (coherentes con (Nx, Nz))
    F_dPdx = np.zeros((Nx, Nz), dtype=np.float64)
    F_dPdz = np.zeros((Nx, Nz), dtype=np.float64)
    F_d2Pdx2 = np.zeros((Nx, Nz), dtype=np.float64)
    F_d2Pdz2 = np.zeros((Nx, Nz), dtype=np.float64)

    one_over_dh = 1.0 / dh

    # bucle temporal
    for t_idx in range(1, Tout):
        # primera aproximación interior (esquinas excluidas)
        # usamos notación P2[ix, iz]
        # interior regular (no-PML): índices ix=1..Nx-2, iz=1..Nz-2
        # computo central para la región interior (vectorizado en bloques)
        # Nota: aquí mantener el esquema original simple
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
        # iteramos desde la derecha hacia la izquierda (coincide con Código B)
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
