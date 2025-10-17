"""
Acoustic wave 2D FDTD (densidad constante) - traducción desde MATLAB a Python

Requiere: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from utils import plot_traces

# ---------------------------
# Parámetros (como en el MATLAB)
# ---------------------------
Tout = 400  # tiempo de propagación (iteraciones de tiempo)
Nx = 200  # puntos en x
Nz = 200  # puntos en z
c = 1200.0  # velocidad de fondo (m/s)
dh = 2.0  # paso espacial (m)
dt = dh / (c * np.sqrt(2))  # paso temporal (condición de Courant)
G = c * dt / dh  # parámetro de Courant

# Matrices de presión: P1 (t-1), P2 (t), P3 (t+1)
P1 = np.zeros((Nz, Nx), dtype=np.float64)
P2 = np.zeros((Nz, Nx), dtype=np.float64)
P3 = np.zeros((Nz, Nx), dtype=np.float64)

alpha = 0.15  # define la anchura del pulso
x0 = dh * Nx / 2
z0 = dh * Nz / 3

# Prealocar "video" (puedes ahorrar memoria guardando menos frames)
# OJO: esto puede consumir mucho RAM cuando Tout es grande
video = np.zeros(
    (Tout, Nz, Nx), dtype=np.float32
)  # guardo como float32 para ahorrar memoria

# ---------------------------
# Fuente inicial (condición inicial) P2
# ---------------------------
# En MATLAB: for i=1:Nz, for j=1:Nx: r2 = (dh*i-z0)^2 + (dh*j-x0)^2
# En Python: índices 0..Nz-1, 0..Nx-1
iz = np.arange(Nz)  # 0..Nz-1
jx = np.arange(Nx)
Z, X = np.meshgrid(iz, jx, indexing="ij")  # Z.shape=(Nz,Nx)
r2 = (dh * (Z + 1) - z0) ** 2 + (dh * (X + 1) - x0) ** 2
# Notar: en MATLAB i comienza en 1, por eso sumo +1 al índice para reproducir exactamente la fórmula original
# P2 = np.sin(1.0 - alpha * r2) * np.exp(-alpha * r2)

# Fuente ricker (temporal)
fq = 20  # frecuencia central (Hz)
t = np.arange(0, Tout * dt, dt)
src = (1 - 2 * (np.pi * fq * (t - 1 / fq)) ** 2) * np.exp(
    -((np.pi * fq * (t - 1 / fq)) ** 2)
)
# plot src para ver la forma del pulso
plt.figure()
plt.plot(t, src)
plt.show()


# Guardar frame 0
video[0, :, :] = P2.astype(np.float32)

# ---------------------------
# Bucle temporal (esquema explícito en diferencias finitas)
# ---------------------------
# Notación: en MATLAB se calculaba P3(i,j) para i=2:Nz-1, j=2:Nx-1
# En Python, índices válidos interiores son 1..Nz-2 y 1..Nx-2
coeff_center = 2.0 - 4.0 * G * G
coeff_neigh = G * G

for t in range(1, Tout):
    # cálculo interior con slicing (vectorizado)
    # P3[1:-1,1:-1] = coeff_center * P2[1:-1,1:-1] +
    #                 coeff_neigh * (P2[2:,1:-1] + P2[:-2,1:-1] + P2[1:-1,2:] + P2[1:-1,:-2]) -
    #                 P1[1:-1,1:-1]
    P3_inner = (
        coeff_center * P2[1:-1, 1:-1]
        + coeff_neigh * (P2[2:, 1:-1] + P2[:-2, 1:-1] + P2[1:-1, 2:] + P2[1:-1, :-2])
        - P1[1:-1, 1:-1]
    )
    P3[1:-1, 1:-1] = P3_inner

    # Actualizar campos: P1 <- P2, P2 <- P3 (sólo en interior)
    # Podemos asignar todo el dominio si queremos:
    P1[:, :] = P2
    P2[:, :] = P3

    # Añadir fuente en cada iteración (en el centro)
    P2[3, Nx // 2] += src[t]
    # Guardar frame t
    video[t, :, :] = P3.astype(np.float32)

    # (opcional) mostrar progreso
    if (t % 100) == 0:
        print(f"Iteración {t}/{Tout}")

# Visualizacion rebenada
print(video.shape)
z_r = 3
shotgather = video[:, z_r, :]
print(shotgather.shape)
plt.figure()
plt.imshow(shotgather, aspect="auto", cmap="seismic")
plt.colorbar(label="Amplitude")
plt.xlabel("Time step")
plt.ylabel("Receiver position (m)")
plt.title(f"Shot gather at surface (z={z_r})")
plt.show()

# Visualización de trazas usando la función plot_traces
plot_traces(
    video,
    z_index=z_r,
    x_label="Receiver Position (m)",
    y_label="Time step",
    title="Shot Gather Traces",
)
# exit()
# ---------------------------
# Visualización (animación)
# ---------------------------

fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(video[0], origin="upper", aspect="auto")
cb = fig.colorbar(im, ax=ax)
ax.set_xlabel("Distance (m)")
ax.set_ylabel("Depth (m)")
title = ax.set_title("Iteración 0")


def update(frame):
    im.set_data(video[frame])
    title.set_text(f"Iteración {frame}")
    return im, title


# Ajusta interval en ms (por ejemplo 10 ms -> similar al pause(0.01) de MATLAB)
ani = FuncAnimation(fig, update, frames=range(0, Tout, 1), interval=10, blit=False)

plt.show()

# Si quieres guardar la animación como mp4 (requiere ffmpeg en el sistema):
# ani.save('wave_simulation.mp4', fps=30, dpi=150)
