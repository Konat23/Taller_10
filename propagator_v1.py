import numpy as np
from numba import jit
from utils import get_CPML
from utils import plot_frame, timing


@jit(nopython=True)
def propagator(m, src, Ix0, Iz0, dx, dz, dt, max_offset, frec):
    max_ix = int(max_offset / dx)
    Nx, Nz = m.shape
    Nt = len(src)

    # Initialize arrays
    P = np.zeros((Nx, Nz, Nt))
    P_Iz0 = np.zeros((Nt, Nx))
    d2P_dt2 = np.zeros((Nx, Nz, Nt))

    # Temporary arrays for CPML
    dP_dx = np.zeros((Nx, Nz))
    dP_dz = np.zeros((Nx, Nz))
    F_dPdx = np.zeros((Nx, Nz))
    F_dPdz = np.zeros((Nx, Nz))
    F_d2Pdx2 = np.zeros((Nx, Nz))
    F_d2Pdz2 = np.zeros((Nx, Nz))

    v2 = m**2

    l_att = 20
    CPMLimit = l_att
    one_over_dx2 = 1.0 / dx**2
    one_over_dz2 = 1.0 / dz**2
    one_over_dx = 1.0 / dx
    one_over_dz = 1.0 / dz

    Vmax = np.max(m)
    Vmin = np.min(m)
    w_l_ = Vmin / frec
    Points_per_wavelength = w_l_ / dx
    Courant_number = Vmax * dt / dx * np.sqrt(2.0)

    # Get CPML coefficients
    R = 1e-3
    Vcpml = Vmax
    a_x, a_x_half, b_x, b_x_half, a_z, a_z_half, b_z, b_z_half = get_CPML(
        CPMLimit, R, Vcpml, Nx, Nz, dx, dz, dt, frec
    )

    for it in range(1, Nt - 1):
        # Snapshot
        P_tmp = P[:, :, it].copy()

        # Free-surface conditions
        d2P_dt2[:, 1, it] = 0  # Note: Python uses 0-based indexing, so iz=2 becomes 1

        # Spatial finite differences - main domain
        for ix in range(
            CPMLimit + 1, Nx - CPMLimit - 1
        ):  # +1 because Python is 0-based
            for iz in range(2, Nz - CPMLimit - 1):  # iz=3 becomes 2
                d2P_dx2 = (
                    P_tmp[ix + 1, iz] + P_tmp[ix - 1, iz] - 2 * P_tmp[ix, iz]
                ) * one_over_dx2
                d2P_dz2 = (
                    P_tmp[ix, iz + 1] + P_tmp[ix, iz - 1] - 2 * P_tmp[ix, iz]
                ) * one_over_dz2
                d2P_dt2[ix, iz, it] = d2P_dx2 + d2P_dz2

        # Boundary conditions
        P_tmp[Nx - 2, :] = 0  # Nx-1 becomes Nx-2
        P_tmp[1, :] = 0  # 2 becomes 1
        P_tmp[:, Nz - 2] = 0  # Nz-1 becomes Nz-2

        # C-PML conditions - Left boundary
        for ix in range(1, CPMLimit + 1):  # 2 becomes 1
            for iz in range(1, Nz - 1):  # 2 becomes 1
                dP_dx[ix, iz] = (P_tmp[ix + 1, iz] - P_tmp[ix, iz]) * one_over_dx
                dP_dz[ix, iz] = (P_tmp[ix, iz + 1] - P_tmp[ix, iz]) * one_over_dz

                F_dPdx[ix, iz] = (
                    F_dPdx[ix, iz] * b_x_half[ix] + a_x_half[ix] * dP_dx[ix, iz]
                )
                dP_dx[ix, iz] = dP_dx[ix, iz] + F_dPdx[ix, iz]

                F_dPdz[ix, iz] = F_dPdz[ix, iz] * b_z[iz] + a_z[iz] * dP_dz[ix, iz]
                dP_dz[ix, iz] = dP_dz[ix, iz] + F_dPdz[ix, iz]

                d2P_dx2 = one_over_dx * (dP_dx[ix, iz] - dP_dx[ix - 1, iz])
                d2P_dz2 = one_over_dz * (dP_dz[ix, iz] - dP_dz[ix, iz - 1])

                F_d2Pdx2[ix, iz] = F_d2Pdx2[ix, iz] * b_x[ix] + a_x[ix] * d2P_dx2
                d2P_dx2 = d2P_dx2 + F_d2Pdx2[ix, iz]

                F_d2Pdz2[ix, iz] = F_d2Pdz2[ix, iz] * b_z[iz] + a_z[iz] * d2P_dz2
                d2P_dz2 = d2P_dz2 + F_d2Pdz2[ix, iz]

                d2P_dt2[ix, iz, it] = d2P_dx2 + d2P_dz2

        # C-PML conditions - Right boundary
        for ix in range(Nx - 2, Nx - CPMLimit - 2, -1):  # Nx-1 becomes Nx-2
            for iz in range(1, Nz - 1):
                dP_dx[ix, iz] = (P_tmp[ix, iz] - P_tmp[ix - 1, iz]) * one_over_dx
                dP_dz[ix, iz] = (P_tmp[ix, iz + 1] - P_tmp[ix, iz]) * one_over_dz

                F_dPdx[ix, iz] = (
                    F_dPdx[ix, iz] * b_x_half[ix - 1] + a_x_half[ix - 1] * dP_dx[ix, iz]
                )
                dP_dx[ix, iz] = dP_dx[ix, iz] + F_dPdx[ix, iz]

                F_dPdz[ix, iz] = F_dPdz[ix, iz] * b_z[iz] + a_z[iz] * dP_dz[ix, iz]
                dP_dz[ix, iz] = dP_dz[ix, iz] + F_dPdz[ix, iz]

                d2P_dx2 = one_over_dx * (dP_dx[ix + 1, iz] - dP_dx[ix, iz])
                d2P_dz2 = one_over_dz * (dP_dz[ix, iz] - dP_dz[ix, iz - 1])

                F_d2Pdx2[ix, iz] = F_d2Pdx2[ix, iz] * b_x[ix] + a_x[ix] * d2P_dx2
                d2P_dx2 = d2P_dx2 + F_d2Pdx2[ix, iz]

                F_d2Pdz2[ix, iz] = F_d2Pdz2[ix, iz] * b_z[iz] + a_z[iz] * d2P_dz2
                d2P_dz2 = d2P_dz2 + F_d2Pdz2[ix, iz]

                d2P_dt2[ix, iz, it] = d2P_dx2 + d2P_dz2

        # C-PML conditions - Bottom boundary
        for ix in range(CPMLimit + 1, Nx - CPMLimit - 1):
            for iz in range(Nz - 2, Nz - CPMLimit - 2, -1):  # Nz-1 becomes Nz-2
                dP_dx[ix, iz] = (P_tmp[ix + 1, iz] - P_tmp[ix, iz]) * one_over_dx
                dP_dz[ix, iz] = (P_tmp[ix, iz] - P_tmp[ix, iz - 1]) * one_over_dz

                F_dPdx[ix, iz] = F_dPdx[ix, iz] * b_x[ix] + a_x[ix] * dP_dx[ix, iz]
                dP_dx[ix, iz] = dP_dx[ix, iz] + F_dPdx[ix, iz]

                F_dPdz[ix, iz] = (
                    F_dPdz[ix, iz] * b_z_half[iz - 1] + a_z_half[iz - 1] * dP_dz[ix, iz]
                )
                dP_dz[ix, iz] = dP_dz[ix, iz] + F_dPdz[ix, iz]

                d2P_dx2 = one_over_dx * (dP_dx[ix, iz] - dP_dx[ix - 1, iz])
                d2P_dz2 = one_over_dz * (dP_dz[ix, iz + 1] - dP_dz[ix, iz])

                F_d2Pdx2[ix, iz] = F_d2Pdx2[ix, iz] * b_x[ix] + a_x[ix] * d2P_dx2
                d2P_dx2 = d2P_dx2 + F_d2Pdx2[ix, iz]

                F_d2Pdz2[ix, iz] = F_d2Pdz2[ix, iz] * b_z[iz] + a_z[iz] * d2P_dz2
                d2P_dz2 = d2P_dz2 + F_d2Pdz2[ix, iz]

                d2P_dt2[ix, iz, it] = d2P_dx2 + d2P_dz2

        # Time integration
        P[:, :, it + 1] = (
            dt**2 * v2 * d2P_dt2[:, :, it] + 2 * P[:, :, it] - P[:, :, it - 1]
        )

        # Source injection
        P[Ix0, Iz0, it + 1] = P[Ix0, Iz0, it + 1] + src[it + 1]

        # Surface component selection
        P_Iz0[it, :] = P[:, Iz0, it]

        if (it % 100) == 0:
            print(f"Iteration {it}/{Tout}")

    # Extract result
    start_ix = max(20, Ix0 - max_ix)  # 21 becomes 20 (0-based)
    end_ix = min(Nx - 21, Ix0 + max_ix)  # Nx-20 becomes Nx-21

    Pt = P[start_ix : end_ix + 1, Iz0, :].T

    return Pt, P, d2P_dt2


if __name__ == "__main__":
    # Example usage
    Tout = 1000
    c = 1200.0
    Nx, Nz = 200, 200
    dh = 2.0
    dx, dz = dh, dh
    dt = dh / (c * np.sqrt(2))
    max_offset = 500.0
    frec = 25.0

    m = np.ones((Nx, Nz)) * c  # Homogeneous medium

    t = np.arange(0, Tout * dt, dt)
    a = (np.pi * frec) ** 2
    t0 = 0.1
    src = -2 * a * (t - t0) * np.exp(-a * (t - t0) ** 2)

    Ix0, Iz0 = Nx // 2, Nz // 2  # Source location

    Pt, P, d2P_dt2 = propagator(m, src, Ix0, Iz0, dx, dz, dt, max_offset, frec)
    print(P.shape)
    print("Propagation complete.")
    # graficar algunos frames (mantengo tu plot_frame)
    frames = [100, 200, 250, 300, 400, 500, 600]
    vmax = 1
    for frame in frames:
        plot_frame(
            P,
            n_save=frame,
            title=f"frame = {frame}",  # vmax=vmax
        )
