"""
Acoustic wave 2D FDTD (constant density) - translated from MATLAB to Python

Requires: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from utils import (
    plot_traces,
    plot_source_wavelet,
    plot_shot_gather,
    animate_video,
)
from numba import jit

# Parameters
Tout = 400
Nx = 200
Nz = 200
c = 1200.0
dh = 2.0
dt = dh / (c * np.sqrt(2))
G = c * dt / dh

alpha = 0.15
x0 = dh * Nx / 2
z0 = dh * Nz / 3

iz = np.arange(Nz)
jx = np.arange(Nx)
Z, X = np.meshgrid(iz, jx, indexing="ij")
r2 = (dh * (Z + 1) - z0) ** 2 + (dh * (X + 1) - x0) ** 2

fq = 20
t = np.arange(0, Tout * dt, dt)
src = (1 - 2 * (np.pi * fq * (t - 1 / fq)) ** 2) * np.exp(
    -((np.pi * fq * (t - 1 / fq)) ** 2)
)

plot_source_wavelet(t, src)


@jit(nopython=True)
def propagate_wave(Tout, Nx, Nz, c, dh, dt, G, src):
    P1 = np.zeros((Nz, Nx), dtype=np.float64)
    P2 = np.zeros((Nz, Nx), dtype=np.float64)
    P3 = np.zeros((Nz, Nx), dtype=np.float64)

    video = np.zeros((Tout, Nz, Nx), dtype=np.float32)
    video[0, :, :] = P2.astype(np.float32)

    coeff_center = 2.0 - 4.0 * G * G
    coeff_neigh = G * G

    for t in range(1, Tout):
        P3_inner = (
            coeff_center * P2[1:-1, 1:-1]
            + coeff_neigh
            * (P2[2:, 1:-1] + P2[:-2, 1:-1] + P2[1:-1, 2:] + P2[1:-1, :-2])
            - P1[1:-1, 1:-1]
        )
        P3[1:-1, 1:-1] = P3_inner

        P1[:, :] = P2
        P2[:, :] = P3

        P2[3, Nx // 2] += src[t]
        video[t, :, :] = P3.astype(np.float32)

        if (t % 100) == 0:
            print(f"Iteration {t}/{Tout}")

    return video


video = propagate_wave(Tout, Nx, Nz, c, dh, dt, G, src)

z_r = 3
plot_shot_gather(video, z_r)

plot_traces(
    video,
    z_index=z_r,
    x_label="Receiver Position (m)",
    y_label="Time step",
    title=f"Shot Gather Traces at z={z_r}",
)

animate_video(video, Tout)
