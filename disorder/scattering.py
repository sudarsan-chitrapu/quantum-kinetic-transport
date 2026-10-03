import numpy as np

from .matrix_elements import (
    disorder_band_basis,
    elastic_scattering_weight
)


def transition_rates(
    disorder,
    k,
    kp,
    energies_k,
    energies_kp,
    eigenvectors_k,
    eigenvectors_kp,
    eta,
    impurity_density=1.0,
    hbar=1.0
):
    # Matrix element for k -> k'
    U_band = disorder_band_basis(
        disorder,
        k,
        kp,
        eigenvectors_k,
        eigenvectors_kp
    )

    n_bands = len(energies_k)

    rates = np.zeros(
        (n_bands, n_bands)
    )

    for m in range(n_bands):
        for n in range(n_bands):

            delta_energy = elastic_scattering_weight(
                energies_k[m],
                energies_kp[n],
                eta
            )

            matrix_element_squared = (
                np.abs(U_band[m, n])**2
            )

            rates[m, n] = (
                2.0 * np.pi
                / hbar
                * impurity_density
                * matrix_element_squared
                * delta_energy
            )

    return rates

def transition_rates_on_mesh(
    disorder,
    energies,
    eigenvectors,
    eta,
    k_weight,
    impurity_density=1.0,
    hbar=1.0
):
    """
    Vectorized transition rates for scalar disorder.

    Output shape:
        (Nk, n_bands, Nk, n_bands)

    W[i, m, j, n] corresponds to
        (m, k_i) -> (n, k_j)
    """

    # <u_m(k_i) | u_n(k_j)>
    overlaps = np.einsum(
        "iam,jan->imjn",
        eigenvectors.conj(),
        eigenvectors
    )

    matrix_element_squared = (
        disorder.strength**2
        * np.abs(overlaps)**2
    )

    # epsilon_m(k_i) - epsilon_n(k_j)
    energy_difference = (
        energies[:, :, None, None]
        - energies[None, None, :, :]
    )

    delta_energy = (
        np.exp(
            -(energy_difference / eta)**2
        )
        / (np.sqrt(np.pi) * eta)
    )

    rates = (
        (2.0 * np.pi / hbar)
        * impurity_density
        * matrix_element_squared
        * delta_energy
        * k_weight
    )

    return rates