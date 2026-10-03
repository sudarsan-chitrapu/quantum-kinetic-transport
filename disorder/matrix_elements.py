import numpy as np


def disorder_band_basis(
    disorder,
    k,
    kp,
    eigenvectors_k,
    eigenvectors_kp
):
    U_orbital = disorder.matrix_element(k, kp)

    U_band = (
        eigenvectors_k.conj().T
        @ U_orbital
        @ eigenvectors_kp
    )

    return U_band

def elastic_scattering_weight(
    energy_initial,
    energy_final,
    eta
):
    energy_difference = (
        energy_initial - energy_final
    )

    return (
        np.exp(
            -(energy_difference / eta)**2
        )
        / (np.sqrt(np.pi) * eta)
    )