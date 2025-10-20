import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


def plot_traces(
    cube_3d,
    cube_3d2=None,
    z_index=0,
    x_label="x",
    y_label="t",
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

    # Plot para cube_3d2 (DeepOnet) si existe
    if cube_3d2 is not None:
        for i in range(Trazas_subset2.shape[0]):
            offset = Rx_subset[i]
            scaled_trace2 = (
                Trazas_subset2[i] / np.max(np.abs(Trazas_subset2[i])) * (Nx / 20)
            )
            label = "DeepOnet" if i == 0 else ""
            plt.plot(
                offset + scaled_trace2,
                tiempo,
                linestyle="dashed",
                color="red",
                alpha=0.7,
                label=label,
            )

    # Ajustar etiquetas e invertir eje Y (tiempo hacia abajo)
    plt.ylabel("Tiempo (s)")  # eje Y ahora es tiempo
    plt.xlabel("Receptores (posición x)")  # eje X ahora es receptores
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


def animate_video(video, Tout, n_save=None):
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
