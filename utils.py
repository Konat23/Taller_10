import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


def plot_traces(
    cube_3d,
    cube_3d2=None,
    z_index=0,
    x_label="Sensor Position (m)",
    y_label="Tiempo (s)",
    title=None,
    cmap="seismic",
    center_colorbar=True,
    max_value=None,
    borde=10,
):
    """
    Plotea trazas de un array 3D para un índice z específico,
    mostrando el tiempo en el eje Y y las trazas en el eje X.
    """

    def get_traces(Rx, Rz, cube_3d):
        NTrazas = Rx.shape[0]
        Nt = cube_3d.shape[0]
        Trazas = np.zeros([NTrazas, Nt])
        for i in range(NTrazas):
            Trazas[i] = cube_3d[:, Rz[i], Rx[i]]
        return Trazas

    Nt, Nx, Nz = cube_3d.shape
    if cube_3d2 is not None:
        assert (
            cube_3d2.shape == cube_3d.shape
        ), "cube_3d2 debe tener la misma forma que cube_3d"

    Rx_all = np.arange(Nx)
    Rz_all = np.full(Rx_all.shape[0], z_index)
    Trazas_all = get_traces(Rx_all, Rz_all, cube_3d)
    if cube_3d2 is not None:
        Trazas_all2 = get_traces(Rx_all, Rz_all, cube_3d2)

    if max_value is not None and center_colorbar:
        raise ValueError("max_value should not be used with center_colorbar=True")

    if max_value is not None:
        vmin, vmax = -max_value, max_value
    else:
        vmin, vmax = Trazas_all.min(), Trazas_all.max()
        if center_colorbar:
            abs_max = max(abs(vmin), abs(vmax))
            vmin, vmax = -abs_max, abs_max

    Rx_subset = np.linspace(borde, Nx - borde - 1, 10, dtype=int)
    print(Rx_subset)
    Rz_subset = np.full(Rx_subset.shape[0], z_index)
    Trazas_subset = get_traces(Rx_subset, Rz_subset, cube_3d)
    if cube_3d2 is not None:
        Trazas_subset2 = get_traces(Rx_subset, Rz_subset, cube_3d2)

    plt.figure(figsize=(5, 6))  # Más alto que ancho

    tiempo = np.arange(Nt)  # eje Y ahora será el tiempo

    # Plot para cube_3d (Referencia) con ejes transpuestos
    for i in range(Trazas_subset.shape[0]):
        offset = Rx_subset[i]
        scaled_trace = Trazas_subset[i] / np.max(np.abs(Trazas_subset[i])) * (Nx / 20)
        label = "Referencia" if i == 0 else ""
        plt.plot(
            offset + scaled_trace, tiempo, color="black", label=label
        )  # X=traza, Y=tiempo

    # Ajustar etiquetas e invertir eje Y (tiempo hacia abajo)
    plt.ylabel(y_label)  # eje Y ahora es tiempo
    plt.xlabel(x_label)  # eje X ahora es receptores
    plt.gca().invert_yaxis()  # Tiempo creciente hacia abajo, típico en sísmica

    plt.title(title if title else f"Trazas en z={z_index}")
    plt.legend(loc="upper right", fontsize="small")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_source_wavelet(t, src):
    plt.figure()
    plt.plot(t, src)
    plt.title("Source Wavelet")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.show()


def plot_shot_gather(video, z_r):
    shotgather = video[:, z_r, :]
    plt.figure()
    plt.imshow(shotgather, aspect="auto", cmap="seismic")
    plt.colorbar(label="Amplitude")
    plt.xlabel("Time step")
    plt.ylabel("Receiver position (m)")
    plt.title(f"Shot gather at surface (z={z_r})")
    plt.show()


def animate_video(video, Tout, n_save=None, vmax=None):
    if vmax is None:
        vmax = np.max(np.abs(video))
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(
        video[0], origin="upper", aspect="auto", cmap="seismic", vmin=-vmax, vmax=vmax
    )
    cb = fig.colorbar(im, ax=ax)
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Depth (m)")
    title = ax.set_title("Iteration 0")

    def update(frame):
        im.set_data(video[frame])
        title.set_text(f"Iteration {frame}")
        return im, title

    ani = FuncAnimation(fig, update, frames=range(0, Tout, 1), interval=10, blit=False)
    plt.show()


def plot_frame(video, n_save=None, vmax=None, title=None):
    T_out = video.shape[0]
    if n_save is None:
        n_save = T_out // 2  # Guardar el cuadro medio si no se especifica
    if n_save < 0 or n_save >= T_out:
        raise ValueError(f"n_save debe estar entre 0 y {T_out - 1}")

    if vmax is None:
        vmax = np.max(np.abs(video[n_save]))
        print(f"vmax no proporcionado. Usando vmax calculado: {vmax}")

    plt.imshow(
        video[n_save],
        origin="upper",
        aspect="auto",
        cmap="seismic",
        vmin=-vmax,
        vmax=vmax,
    )
    plt.colorbar(label="Amplitude")
    plt.xlabel("Distance (m)")
    plt.ylabel("Depth (m)")
    plt.title(f"{title if title else f'Frame at iteration {n_save}'}")
    plt.savefig(f"frame_{n_save}.png")
    print(f"Frame {n_save} saved as 'frame_{n_save}.png'")

    plt.show()


from functools import wraps
import time
import sys


def timing(f):
    """Decorator for measuring the execution time of methods."""

    @wraps(f)
    def wrapper(*args, **kwargs):
        ts = time.time()
        result = f(*args, **kwargs)
        te = time.time()
        elapsed = te - ts
        print("%r took %f s\n" % (f.__name__, elapsed))
        sys.stdout.flush()
        wrapper.elapsed_time = elapsed  # Guarda el tiempo en un atributo
        return result

    wrapper.elapsed_time = None  # Inicializa el atributo
    return wrapper


from numba import jit


@jit(nopython=True)
def get_CPML(CPMLimit, R, Vcpml, Nx, Nz, dx, dz, dt, frec):
    """
    Implementación Python corregida - inicialización con ceros como en MATLAB
    """
    # CORREGIDO: Inicializar con CEROS como en MATLAB
    a_x = np.zeros(Nx, dtype=np.float64)
    a_x_half = np.zeros(Nx, dtype=np.float64)
    b_x = np.zeros(Nx, dtype=np.float64)  # CEROS, no unos
    b_x_half = np.zeros(Nx, dtype=np.float64)  # CEROS, no unos

    a_z = np.zeros(Nz, dtype=np.float64)
    a_z_half = np.zeros(Nz, dtype=np.float64)
    b_z = np.zeros(Nz, dtype=np.float64)  # CEROS, no unos
    b_z_half = np.zeros(Nz, dtype=np.float64)  # CEROS, no unos

    # Parámetros PML
    D_pml_x = CPMLimit * dx
    D_pml_z = CPMLimit * dz
    d0_x = -3.0 / (2.0 * D_pml_x) * np.log(R)
    d0_z = -3.0 / (2.0 * D_pml_z) * np.log(R)

    # Arrays temporales
    x = np.zeros(CPMLimit + 1)
    x_half = np.zeros(CPMLimit + 1)
    alpha_x = np.zeros(CPMLimit + 1)
    alpha_x_half = np.zeros(CPMLimit + 1)

    z = np.zeros(CPMLimit + 1)
    z_half = np.zeros(CPMLimit + 1)
    alpha_z = np.zeros(CPMLimit + 1)
    alpha_z_half = np.zeros(CPMLimit + 1)

    # Inicializar perfiles PML (igual que antes)
    for j in range(CPMLimit + 1):
        x[j] = (CPMLimit - j) * dx
        z[j] = (CPMLimit - j) * dz
        alpha_x[j] = np.pi * frec * (D_pml_x - x[j]) / D_pml_x
        alpha_z[j] = np.pi * frec * (D_pml_z - z[j]) / D_pml_z

        x_half[j] = (CPMLimit - j) * dx - dx / 2.0
        z_half[j] = (CPMLimit - j) * dz - dz / 2.0
        alpha_x_half[j] = np.pi * frec * (D_pml_x - x_half[j]) / D_pml_x
        alpha_z_half[j] = np.pi * frec * (D_pml_z - z_half[j]) / D_pml_z

    # Bottom side (borde inferior)
    for j in range(Nz - CPMLimit - 1, Nz):
        idx = Nz - j - 1  # Índice para arrays PML
        if idx < len(z):
            d_z_val = d0_z * Vcpml * ((z[idx] / D_pml_z) ** 2)
            b_z[j] = np.exp(-(d_z_val + alpha_z[idx]) * dt)
            if np.abs(d_z_val + alpha_z[idx]) > 1e-20:
                a_z[j] = d_z_val / (d_z_val + alpha_z[idx]) * (b_z[j] - 1.0)
            else:
                a_z[j] = 0.0

            # Para arrays _half
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

    # Left side (borde izquierdo)
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

    # Right side (borde derecho)
    for i in range(Nx - CPMLimit - 1, Nx):
        idx = Nx - i - 1  # Índice para arrays PML
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
