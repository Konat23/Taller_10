import matplotlib.pyplot as plt
import numpy as np


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

    plt.figure(figsize=(5, 6))  # 🔄 Más alto que ancho

    tiempo = np.arange(Nt)  # 🔄 eje Y ahora será el tiempo

    # 🔄 Plot para cube_3d (Referencia) con ejes transpuestos
    for i in range(Trazas_subset.shape[0]):
        offset = Rx_subset[i]
        scaled_trace = Trazas_subset[i] / np.max(np.abs(Trazas_subset[i])) * (Nx / 20)
        label = "Referencia" if i == 0 else ""
        plt.plot(
            offset + scaled_trace, tiempo, color="black", label=label
        )  # 🔄 X=traza, Y=tiempo

    # 🔄 Plot para cube_3d2 (DeepOnet) si existe
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

    # 🔄 Ajustar etiquetas e invertir eje Y (tiempo hacia abajo)
    plt.ylabel("Tiempo (s)")  # 🔄 eje Y ahora es tiempo
    plt.xlabel("Receptores (posición x)")  # 🔄 eje X ahora es receptores
    plt.gca().invert_yaxis()  # 🔄 Tiempo creciente hacia abajo, típico en sísmica

    plt.title(title if title else f"Trazas en z={z_index}")
    plt.legend(loc="upper right", fontsize="small")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
