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
def get_CPML(CPMLimit, R, Vcpml, Nx, Nz, dx, dz, dt, frec):
    """
    Calcula coeficientes para CPML coherente con implementación MATLAB

    Parameters:
    CPMLimit: Grosor de la CPML en número de celdas
    R: Coeficiente de reflexión deseado
    Vcpml: Velocidad para escalado (usualmente velocidad máxima)
    Nx, Nz: Dimensiones de la malla
    dx, dz: Espaciamientos espaciales
    dt: Paso temporal
    frec: Frecuencia de referencia
    """

    # Inicializar arrays
    a_x = np.zeros(Nx, dtype=np.float64)
    a_x_half = np.zeros(Nx, dtype=np.float64)
    b_x = np.ones(Nx, dtype=np.float64)
    b_x_half = np.ones(Nx, dtype=np.float64)

    a_z = np.zeros(Nz, dtype=np.float64)
    a_z_half = np.zeros(Nz, dtype=np.float64)
    b_z = np.ones(Nz, dtype=np.float64)
    b_z_half = np.ones(Nz, dtype=np.float64)

    # Calcular espesores físicos y amortiguamiento máximo
    D_pml_x = CPMLimit * dx
    D_pml_z = CPMLimit * dz
    d0_x = -3.0 / (2.0 * D_pml_x) * np.log(R)
    d0_z = -3.0 / (2.0 * D_pml_z) * np.log(R)

    # Pre-calcular coordenadas y alpha values como en MATLAB
    x = np.zeros(CPMLimit + 1)
    x_half = np.zeros(CPMLimit + 1)
    alpha_x = np.zeros(CPMLimit + 1)
    alpha_x_half = np.zeros(CPMLimit + 1)

    z = np.zeros(CPMLimit + 1)
    z_half = np.zeros(CPMLimit + 1)
    alpha_z = np.zeros(CPMLimit + 1)
    alpha_z_half = np.zeros(CPMLimit + 1)

    for j in range(CPMLimit + 1):
        # Coordenadas para posiciones enteras
        x[j] = (CPMLimit - j) * dx
        z[j] = (CPMLimit - j) * dz
        alpha_x[j] = np.pi * frec * (D_pml_x - x[j]) / D_pml_x
        alpha_z[j] = np.pi * frec * (D_pml_z - z[j]) / D_pml_z

        # Coordenadas para posiciones half (desplazadas media celda)
        x_half[j] = (CPMLimit - j) * dx - dx / 2.0
        z_half[j] = (CPMLimit - j) * dz - dz / 2.0
        alpha_x_half[j] = np.pi * frec * (D_pml_x - x_half[j]) / D_pml_x
        alpha_z_half[j] = np.pi * frec * (D_pml_z - z_half[j]) / D_pml_z

    # === BORDE INFERIOR (Bottom side) ===
    for j in range(Nz - CPMLimit - 1, Nz):
        idx = Nz - j - 1  # Índice invertido para mirroring

        if idx < len(z):
            # Posiciones enteras
            d_z_val = d0_z * Vcpml * ((z[idx] / D_pml_z) ** 2)
            b_z[j] = np.exp(-(d_z_val + alpha_z[idx]) * dt)
            if np.abs(d_z_val + alpha_z[idx]) > 1e-20:
                a_z[j] = d_z_val / (d_z_val + alpha_z[idx]) * (b_z[j] - 1.0)
            else:
                a_z[j] = 0.0

            # Posiciones half (una celda antes)
            if j == Nz - CPMLimit - 1:
                # Borde de la región PML - valores cero
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

    # === BORDE IZQUIERDO (Left side) ===
    for i in range(CPMLimit + 1):
        if i < len(x):
            # Posiciones enteras
            d_x_val = d0_x * Vcpml * ((x[i] / D_pml_x) ** 2)
            b_x[i] = np.exp(-(d_x_val + alpha_x[i]) * dt)
            if np.abs(d_x_val + alpha_x[i]) > 1e-20:
                a_x[i] = d_x_val / (d_x_val + alpha_x[i]) * (b_x[i] - 1.0)
            else:
                a_x[i] = 0.0

            # Posiciones half
            if i == CPMLimit:
                # Borde de la región PML - valores cero
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

    # === BORDE DERECHO (Right side) ===
    for i in range(Nx - CPMLimit - 1, Nx):
        idx = Nx - i - 1  # Índice invertido para mirroring

        if idx < len(x):
            # Posiciones enteras
            d_x_val = d0_x * Vcpml * ((x[idx] / D_pml_x) ** 2)
            b_x[i] = np.exp(-(d_x_val + alpha_x[idx]) * dt)
            if np.abs(d_x_val + alpha_x[idx]) > 1e-20:
                a_x[i] = d_x_val / (d_x_val + alpha_x[idx]) * (b_x[i] - 1.0)
            else:
                a_x[i] = 0.0

            # Posiciones half (una celda antes)
            if i == Nx - CPMLimit - 1:
                # Borde de la región PML - valores cero
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
    fq = 25.0
    Ix0 = [Nx // 2]
    Iz0 = Nz // 2

    a_x, a_x_half, b_x, b_x_half, a_z, a_z_half, b_z, b_z_half = get_CPML(
        CPMLimit, R, Vmax, Nx, Nz, dh, dh, dt, fq
    )

    F_dPdx = np.zeros((Nz, Nx), dtype=np.float64)
    F_dPdz = np.zeros((Nz, Nx), dtype=np.float64)
    F_d2Pdx2 = np.zeros((Nz, Nx), dtype=np.float64)
    F_d2Pdz2 = np.zeros((Nz, Nx), dtype=np.float64)

    one_over_dh = 1.0 / dh
    one_over_dh2 = 1.0 / (dh * dh)

    for t_idx in range(1, Tout):
        P3_inner = (
            coeff_center * P2[1:-1, 1:-1]
            + coeff_neigh
            * (P2[2:, 1:-1] + P2[:-2, 1:-1] + P2[1:-1, 2:] + P2[1:-1, :-2])
            - P1[1:-1, 1:-1]
        )
        P3[1:-1, 1:-1] = P3_inner

        for ix in range(1, CPMLimit + 1):
            for iz in range(1, Nz - 1):
                dP_dx = (P2[iz, ix + 1] - P2[iz, ix]) * one_over_dh
                dP_dz = (P2[iz + 1, ix] - P2[iz, ix]) * one_over_dh

                F_dPdx[iz, ix] = F_dPdx[iz, ix] * b_x_half[ix] + a_x_half[ix] * dP_dx
                dP_dx_mod = dP_dx + F_dPdx[iz, ix]

                F_dPdz[iz, ix] = F_dPdz[iz, ix] * b_z[iz] + a_z[iz] * dP_dz
                dP_dz_mod = dP_dz + F_dPdz[iz, ix]

                d2P_dx2 = (
                    dP_dx_mod - (P2[iz, ix] - P2[iz, ix - 1]) * one_over_dh
                ) * one_over_dh
                d2P_dz2 = (
                    dP_dz_mod - (P2[iz, ix] - P2[iz - 1, ix]) * one_over_dh
                ) * one_over_dh

                F_d2Pdx2[iz, ix] = F_d2Pdx2[iz, ix] * b_x[ix] + a_x[ix] * d2P_dx2
                d2P_dx2_mod = d2P_dx2 + F_d2Pdx2[iz, ix]

                F_d2Pdz2[iz, ix] = F_d2Pdz2[iz, ix] * b_z[iz] + a_z[iz] * d2P_dz2
                d2P_dz2_mod = d2P_dz2 + F_d2Pdz2[iz, ix]

                laplacian_mod = d2P_dx2_mod + d2P_dz2_mod
                P3[iz, ix] = (
                    c * c * laplacian_mod * dt * dt + 2 * P2[iz, ix] - P1[iz, ix]
                )

        for ix in range(Nx - CPMLimit - 1, Nx - 1):
            for iz in range(1, Nz - 1):
                dP_dx = (P2[iz, ix] - P2[iz, ix - 1]) * one_over_dh
                dP_dz = (P2[iz + 1, ix] - P2[iz, ix]) * one_over_dh

                F_dPdx[iz, ix] = (
                    F_dPdx[iz, ix] * b_x_half[ix - 1] + a_x_half[ix - 1] * dP_dx
                )
                dP_dx_mod = dP_dx + F_dPdx[iz, ix]

                F_dPdz[iz, ix] = F_dPdz[iz, ix] * b_z[iz] + a_z[iz] * dP_dz
                dP_dz_mod = dP_dz + F_dPdz[iz, ix]

                d2P_dx2 = (
                    (P2[iz, ix + 1] - P2[iz, ix]) * one_over_dh - dP_dx_mod
                ) * one_over_dh
                d2P_dz2 = (
                    dP_dz_mod - (P2[iz, ix] - P2[iz - 1, ix]) * one_over_dh
                ) * one_over_dh

                F_d2Pdx2[iz, ix] = F_d2Pdx2[iz, ix] * b_x[ix] + a_x[ix] * d2P_dx2
                d2P_dx2_mod = d2P_dx2 + F_d2Pdx2[iz, ix]

                F_d2Pdz2[iz, ix] = F_d2Pdz2[iz, ix] * b_z[iz] + a_z[iz] * d2P_dz2
                d2P_dz2_mod = d2P_dz2 + F_d2Pdz2[iz, ix]

                laplacian_mod = d2P_dx2_mod + d2P_dz2_mod
                P3[iz, ix] = (
                    c * c * laplacian_mod * dt * dt + 2 * P2[iz, ix] - P1[iz, ix]
                )

        for ix in range(CPMLimit + 1, Nx - CPMLimit - 1):
            for iz in range(Nz - CPMLimit - 1, Nz - 1):
                dP_dx = (P2[iz, ix + 1] - P2[iz, ix]) * one_over_dh
                dP_dz = (P2[iz, ix] - P2[iz - 1, ix]) * one_over_dh

                F_dPdx[iz, ix] = F_dPdx[iz, ix] * b_x[ix] + a_x[ix] * dP_dx
                dP_dx_mod = dP_dx + F_dPdx[iz, ix]

                F_dPdz[iz, ix] = (
                    F_dPdz[iz, ix] * b_z_half[iz - 1] + a_z_half[iz - 1] * dP_dz
                )
                dP_dz_mod = dP_dz + F_dPdz[iz, ix]

                d2P_dx2 = (
                    dP_dx_mod - (P2[iz, ix] - P2[iz, ix - 1]) * one_over_dh
                ) * one_over_dh
                d2P_dz2 = (
                    (P2[iz + 1, ix] - P2[iz, ix]) * one_over_dh - dP_dz_mod
                ) * one_over_dh

                F_d2Pdx2[iz, ix] = F_d2Pdx2[iz, ix] * b_x[ix] + a_x[ix] * d2P_dx2
                d2P_dx2_mod = d2P_dx2 + F_d2Pdx2[iz, ix]

                F_d2Pdz2[iz, ix] = F_d2Pdz2[iz, ix] * b_z[iz] + a_z[iz] * d2P_dz2
                d2P_dz2_mod = d2P_dz2 + F_d2Pdz2[iz, ix]

                laplacian_mod = d2P_dx2_mod + d2P_dz2_mod
                P3[iz, ix] = (
                    c * c * laplacian_mod * dt * dt + 2 * P2[iz, ix] - P1[iz, ix]
                )

        P1[:, :] = P2
        P2[:, :] = P3

        P2[Iz0, Ix0[0]] += src[t_idx, 0]
        video[t_idx, :, :] = P3.astype(np.float32)

        if (t_idx % 100) == 0:
            print(f"Iteration {t_idx}/{Tout}")

    return video


video = propagate_wave_cpml(Tout, Nx, Nz, c, dh, dt, G, src)


frames = [100, 150, 200, 250, 300]
vmax = np.max(np.abs(video[100]))
for frame in frames:
    plot_frame(
        video, n_save=frame, title=f"Δt = {dt:.6e} s, frame = {frame}", vmax=vmax
    )
