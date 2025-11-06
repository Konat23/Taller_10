import numpy as np
import matplotlib.pyplot as plt
from numba import jit

# (asumes que tienes utils.py con las funciones de plotting)
from utils import (
    plot_traces,
    plot_source_wavelet,
    plot_shot_gather,
    animate_video,
    plot_frame,
)
from utils import timing

# ------------------------
# Parámetros
# ------------------------
Tout = 400
Nx = 200
Nz = 200
c = 1200.0  #
dh = 2.0
dt = dh / (c * np.sqrt(2))
print(f"dt inicial: {dt}")
alpha = 0.15

# Índices de fuente EN ÍNDICES (no en metros)
Ix0 = np.array([Nx // 2], dtype=np.int64)  # centro en x (índice)
Iz0 = Nz // 3  # índice z

# coordenadas físicas si las quieres (no usadas en la simulación numérica)
x0 = dh * (Ix0[0] + 1)
z0 = dh * (Iz0 + 1)

# mallado auxiliar
iz = np.arange(Nz)
jx = np.arange(Nx)
Z, X = np.meshgrid(iz, jx, indexing="ij")
r2 = (dh * (Z + 1) - z0) ** 2 + (dh * (X + 1) - x0) ** 2

# fuente (Ricker-like)
fq = 20.0
t = np.arange(0, Tout * dt, dt)
src1d = (1 - 2 * (np.pi * fq * (t - 1.0 / fq)) ** 2) * np.exp(
    -((np.pi * fq * (t - 1.0 / fq)) ** 2)
)

# forzamos que src tenga forma (Nt, nsrc). Aquí nsrc=1
src = src1d.reshape(-1, 1)  # shape (Nt, 1)

# velocidad (modelo simple)
vp_ori = np.zeros((Nx, Nz), dtype=np.float64)
for iz_idx in range(Nz):
    vp_ori[:, iz_idx] = c  # + 0.7 * dh * iz_idx


# ------------------------
# CPML coefficients
# ------------------------
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
    # d0 coefficients (attenuation strength)
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
            # espejo para el borde derecho (coeficientes iguales a los de la distancia al borde)
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
            # espejo para el fondo (abajo)
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


# ------------------------
# Propagador
# ------------------------
@jit(nopython=True)
def propagate_CPML(m, src, Ix0, Iz0, dx, dz, dt, max_offset, f0):
    """
    m: (Nx, Nz) velocity (float64)
    src: (Nt, nsrc) fuente(s) MUST be 2D
    Ix0: 1D array int64 with x-indices of sources (length nsrc)
    Iz0: int scalar (z-index)
    dx, dz, dt: floats
    max_offset, f0: (no usados en este código, f0 se pasa al CPML)
    """
    Nx, Nz = m.shape
    Nt = src.shape[0]
    nsrc = src.shape[1]

    v2 = m * m
    CPMLimit = 20  # ancho CPML en puntos (ajusta si quieres)

    # prealocación (nota: esto ocupa mucha memoria; para producción usa rotación temporal)
    P = np.zeros((Nx, Nz, Nt), dtype=np.float64)
    d2P_dt2 = np.zeros((Nx, Nz, Nt), dtype=np.float64)
    dP_dx = np.zeros((Nx, Nz, Nt), dtype=np.float64)
    dP_dz = np.zeros((Nx, Nz, Nt), dtype=np.float64)
    F_dPdx = np.zeros((Nx, Nz, Nt), dtype=np.float64)
    F_dPdz = np.zeros((Nx, Nz, Nt), dtype=np.float64)

    Vmax = np.max(m)
    R = 1e-3

    a_x, a_x_half, b_x, b_x_half, a_z, a_z_half, b_z, b_z_half = get_CPML(
        CPMLimit, R, Vmax, Nx, Nz, dx, dz, dt, f0
    )

    one_over_dx = 1.0 / dx
    one_over_dz = 1.0 / dz
    one_over_dx2 = one_over_dx * one_over_dx
    one_over_dz2 = one_over_dz * one_over_dz

    # time stepping (leapfrog-like)
    for it in range(1, Nt - 1):
        # slice actual (usado como lectura)
        P_tmp = P[:, :, it]

        # calculamos derivadas centrales en todo el dominio interior (1..Nx-2, 1..Nz-2)
        # y aplicamos CPML en izquierda, derecha y abajo (no arriba)
        for ix in range(1, Nx - 1):
            for iz_idx in range(1, Nz - 1):
                # derivadas centrales
                dpx = (
                    (P_tmp[ix + 1, iz_idx] - P_tmp[ix - 1, iz_idx]) * 0.5 * one_over_dx
                )
                dpz = (
                    (P_tmp[ix, iz_idx + 1] - P_tmp[ix, iz_idx - 1]) * 0.5 * one_over_dz
                )

                # coeficientes CPML en x (izquierda, derecha)
                if ix < CPMLimit:
                    bxh = b_x_half[ix]
                    axh = a_x_half[ix]
                elif ix >= Nx - CPMLimit:
                    idx_mirror = Nx - 1 - ix
                    bxh = b_x_half[idx_mirror]
                    axh = a_x_half[idx_mirror]
                else:
                    bxh = 1.0
                    axh = 0.0

                # coeficientes CPML en z (solo abajo; no aplicamos CPML arriba)
                if iz_idx < CPMLimit:
                    bzh = b_z_half[iz_idx]
                    azh = a_z_half[iz_idx]
                elif iz_idx >= Nz - CPMLimit:
                    idx_mirror_z = Nz - 1 - iz_idx
                    bzh = b_z_half[idx_mirror_z]
                    azh = a_z_half[idx_mirror_z]
                else:
                    bzh = 1.0
                    azh = 0.0

                # actualizar memorias CPML semientradas
                F_dPdx[ix, iz_idx, it] = F_dPdx[ix, iz_idx, it] * bxh + axh * dpx
                dpx += F_dPdx[ix, iz_idx, it]

                F_dPdz[ix, iz_idx, it] = F_dPdz[ix, iz_idx, it] * bzh + azh * dpz
                dpz += F_dPdz[ix, iz_idx, it]

                # guardar derivadas modificadas
                dP_dx[ix, iz_idx, it] = dpx
                dP_dz[ix, iz_idx, it] = dpz

        # ahora construimos la laplaciana (segunda derivadas) a partir de las derivadas modificadas
        # (usamos diferencias hacia atrás/adelante de dP_dx y dP_dz)
        for ix in range(1, Nx - 1):
            for iz_idx in range(1, Nz - 1):
                d2P_dx2 = (
                    (dP_dx[ix + 1, iz_idx, it] - dP_dx[ix - 1, iz_idx, it])
                    * 0.5
                    * one_over_dx
                )
                d2P_dz2 = (
                    (dP_dz[ix, iz_idx + 1, it] - dP_dz[ix, iz_idx - 1, it])
                    * 0.5
                    * one_over_dz
                )
                d2P_dt2[ix, iz_idx, it] = d2P_dx2 + d2P_dz2

        # multiplicar por v^2 y actualizar campo P
        for ix in range(1, Nx - 1):
            for iz_idx in range(1, Nz - 1):
                d2P_dt2[ix, iz_idx, it] *= v2[ix, iz_idx]

        for ix in range(1, Nx - 1):
            for iz_idx in range(1, Nz - 1):
                P[ix, iz_idx, it + 1] = (
                    2.0 * P[ix, iz_idx, it]
                    - P[ix, iz_idx, it - 1]
                    + (dt * dt) * d2P_dt2[ix, iz_idx, it]
                )

        # Condiciones simples en frontera (puedes mejorar si quieres): fijamos bordes a cero
        # esto evita índices fuera de rango cuando inyectamos fuente cerca del borde
        P[0, :, it + 1] = 0.0
        P[Nx - 1, :, it + 1] = 0.0
        P[:, 0, it + 1] = 0.0
        P[:, Nz - 1, it + 1] = 0.0

        # inyectar fuentes (soporta múltiples fuentes)
        for s_idx in range(Ix0.shape[0]):
            sx = Ix0[s_idx]
            if sx >= 0 and sx < Nx and Iz0 >= 0 and Iz0 < Nz:
                P[sx, Iz0, it + 1] += src[it + 1, s_idx]

    # armar shot gather Pt: (Nt, nsrc)
    Pt = np.zeros((Nt, Ix0.shape[0]), dtype=np.float64)
    for i in range(Ix0.shape[0]):
        sx = Ix0[i]
        for it in range(Nt):
            Pt[it, i] = P[sx, Iz0, it]

    return Pt, P, d2P_dt2


# ------------------------
# Ejecutar propagación
# ------------------------
Pt, video, d2 = propagate_CPML(vp_ori, src, Ix0, Iz0, dh, dh, dt, 0, fq)
video = np.transpose(video, (2, 0, 1))  # (Nt, Nx, Nz)
print(f"Shape of video {video.shape}")
# ------------------------
# Visualizaciones (ejemplo)
# ------------------------
z_r = 3
# plot_traces(video, z_index=z_r, title=f"z={z_r}")
print("max absoluto en video:", np.max(video))
print(video.shape)
# mostrar un frame concreto (ajusta frame si quieres)
frame = 200
title = f"Δt = {dt:.6e} s, frame = {frame}"
plot_frame(video, n_save=frame, title=title, vmax=0.1)
