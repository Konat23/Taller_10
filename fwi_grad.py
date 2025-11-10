import numpy as np
from propagator_v1 import propagator


def FWI_GRAD(
    x, Nx, Nz, Nt, g1, Sx1, Sz, dx, dz, dt, offset_max, frec, Pt_obs, k, cpml_size=20
):
    print("Propagating forward wave...")
    Sx1 = np.asarray(Sx1, dtype=np.int64).reshape(-1)
    Pt_mod, P_mod, d2P_dt2 = propagator(
        x.reshape(Nx, Nz), g1, Sx1, Sz, dx, dz, dt, offset_max, frec
    )
    print(f"P_mod shape: {P_mod.shape}")

    if k < 3:
        ad = 5
    elif k == 3 or k == 5:
        ad = 1
    else:
        ad = 0

    diff = Pt_mod[: Nt * (Nx - 40 - ad)].reshape(-1, 1) - Pt_obs[
        : Nt * (Nx - 40 - ad)
    ].reshape(-1, 1)
    f = 0.5 * np.dot(diff.T, diff)

    res = Pt_mod - Pt_obs
    print("Propagating backwave...")
    Pt_back, P_back, d2P_dt2_Back = propagator(
        x.reshape(Nx, Nz),
        np.flipud(res),
        np.arange(cpml_size, Nx - cpml_size),
        Sz,
        dx,
        dz,
        dt,
        offset_max,
        frec,
        cpml_size=20,
    )
    print(f"P_back shape: {P_back.shape}")

    P_back_t = np.zeros((Nx, Nz, Nt))
    for it in range(Nt):
        P_back_t[:, :, it] = P_back[:, :, Nt - it - 1]

    gradient = -dt * np.sum(P_back_t * d2P_dt2, axis=2)
    gradient[Sx1, Sz] = gradient[Sx1 - 1, Sz]
    gradient[Sx1, Sz + 1] = gradient[Sx1 - 1, Sz]
    gradient = gradient * (1 / np.linalg.norm(gradient)) * 100

    g = gradient.reshape(Nx * Nz, 1)
    return f.item(), g


if __name__ == "__main__":
    import numpy as np

    Nx, Nz, Nt = 100, 80, 201
    dx, dz, dt = 10.0, 10.0, 0.001
    offset_max = 500
    frec = 25
    Sx1, Sz = 50, 40
    g1 = np.exp(-((np.arange(Nt) - 50) ** 2) / (2 * 10**2))
    x = np.ones((Nx * Nz,))
    Pt_obs = np.random.randn(Nt, Nx - 40)
    k = 2

    f, g = FWI_GRAD(x, Nx, Nz, Nt, g1, Sx1, Sz, dx, dz, dt, offset_max, frec, Pt_obs, k)

    print("Función objetivo f:", f)
    print("Gradiente g shape:", g.shape)
    print("Norma del gradiente:", np.linalg.norm(g))
