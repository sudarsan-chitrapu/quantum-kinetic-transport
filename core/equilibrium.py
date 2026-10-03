import numpy as np


def fermi_dirac(energies, mu, temperature, kB=1.0):

    if temperature == 0:
        return np.where(
            energies < mu,
            1.0,
            np.where(energies > mu, 0.0, 0.5)
        )

    x = (energies - mu) / (kB * temperature)

    return 1.0 / (np.exp(x) + 1.0)

def fermi_dirac_derivative(
    energies,
    mu,
    temperature,
    kB=1.0
):
    if temperature <= 0:
        raise ValueError(
            "fermi_dirac_derivative requires temperature > 0."
        )

    f0 = fermi_dirac(
        energies,
        mu,
        temperature,
        kB
    )

    return -f0 * (1.0 - f0) / (kB * temperature)

def delta_gaussian(energies, mu, eta):
    """
    Gaussian approximation to delta(energy - mu).

    eta controls the energy broadening.
    """
    x = (energies - mu) / eta

    return (
        np.exp(-x**2)
        / (np.sqrt(np.pi) * eta)
    )