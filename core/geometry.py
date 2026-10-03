import numpy as np


def velocity_operators(model, k):

    dH = model.dH_dk(k)

    velocities = dH / model.hbar

    return velocities

def velocity_band_basis(model, k, eigenvectors):

    velocities = velocity_operators(model, k)

    U = eigenvectors
    U_dagger = U.conj().T

    velocities_band = []

    for v in velocities:
        v_band = U_dagger @ v @ U
        velocities_band.append(v_band)

    return np.array(velocities_band)

def berry_connection_offdiagonal( energies, velocities_band, hbar=1.0, degeneracy_tol=1e-10):

    dimension = velocities_band.shape[0]
    n_bands = len(energies)

    A = np.zeros(
        (dimension, n_bands, n_bands),
        dtype=complex
    )

    for alpha in range(dimension):

        for m in range(n_bands):

            for n in range(n_bands):

                if m != n:

                    energy_difference = (
                        energies[n] - energies[m]
                    )

                    if abs(energy_difference) < degeneracy_tol:
                        A[alpha, m, n] = np.nan

                    else:
                        A[alpha, m, n] = (
                            1j
                            * hbar
                            * velocities_band[alpha, m, n]
                            / energy_difference
                        )

    return A

def geometry_on_mesh(model, k_points, all_energies, all_eigenvectors):
    
    all_velocities_band = []
    all_berry_connections = []

    for i, k in enumerate(k_points):

        energies = all_energies[i]
        eigenvectors = all_eigenvectors[i]

        velocities_band = velocity_band_basis(
            model,
            k,
            eigenvectors
        )

        berry_connection = berry_connection_offdiagonal(
            energies,
            velocities_band,
            hbar=model.hbar
        )

        all_velocities_band.append(velocities_band)
        all_berry_connections.append(berry_connection)

    return (
        np.array(all_velocities_band),
        np.array(all_berry_connections)
    )

def berry_curvature_2d(berry_connection):
            
    n_bands = berry_connection.shape[1]

    curvature = np.zeros(n_bands)

    Ax = berry_connection[0]
    Ay = berry_connection[1]

    for n in range(n_bands):

        for m in range(n_bands):

            if m != n:

                curvature[n] += (
                    -2.0
                    * np.imag(
                        Ax[n, m] * Ay[m, n]
                    )
                )

    return curvature

def berry_curvature_on_mesh(all_berry_connections):

    all_curvatures = []

    for berry_connection in all_berry_connections:

        curvature = berry_curvature_2d(
            berry_connection
        )

        all_curvatures.append(curvature)

    return np.array(all_curvatures)

