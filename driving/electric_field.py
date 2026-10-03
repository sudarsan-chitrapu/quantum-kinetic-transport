import numpy as np

from core.equilibrium import delta_gaussian


def diagonal_electric_drive(
    energies,
    velocities_band,
    electric_field,
    mu,
    eta,
    charge=1.0
):
    """
    T = 0 diagonal electric-field driving term.
    """

    # Numerical representation of delta(epsilon - mu)
    delta_F = delta_gaussian(
        energies,
        mu,
        eta
    )

    # Diagonal band velocities:
    # shape -> (Nk, dimension, n_bands)
    diagonal_velocities = np.diagonal(
        velocities_band,
        axis1=2,
        axis2=3
    ).real

    # E dot v_n(k)
    E_dot_v = np.einsum(
        "d,kdn->kn",
        electric_field,
        diagonal_velocities
    )

    # df0/depsilon = -delta(epsilon - mu)
    drive = (
        -charge
        * E_dot_v
        * delta_F
    )

    return drive