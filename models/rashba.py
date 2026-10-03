import numpy as np

from .base import BlochModel


class RashbaModel(BlochModel):

    def __init__(
        self,
        effective_mass,
        alpha,
        magnetization=0.0,
        hbar=1.0
    ):
        super().__init__(dimension=2, n_bands=2)

        self.effective_mass = effective_mass
        self.alpha = alpha
        self.magnetization = magnetization
        self.hbar = hbar

    def H(self, k):

        kx, ky = k

        kinetic_energy = (
            self.hbar**2 * (kx**2 + ky**2)
            / (2 * self.effective_mass)
        )

        sigma_x = np.array([
            [0, 1],
            [1, 0]
        ], dtype=complex)

        sigma_y = np.array([
            [0, -1j],
            [1j, 0]
        ], dtype=complex)

        sigma_z = np.array([
            [1, 0],
            [0, -1]
        ], dtype=complex)

        identity = np.eye(2, dtype=complex)

        H_kinetic = kinetic_energy * identity

        H_rashba = self.alpha * (
            ky * sigma_x - kx * sigma_y
        )

        H_magnetic = (
            self.magnetization * sigma_z
        )

        return (    
            H_kinetic
            + H_rashba
            + H_magnetic
        )

    def dH_dk(self, k):

        kx, ky = k

        sigma_x = np.array([
            [0, 1],
            [1, 0]
        ], dtype=complex)

        sigma_y = np.array([
            [0, -1j],
            [1j, 0]
        ], dtype=complex)

        identity = np.eye(2, dtype=complex)

        dH_dkx = (
            (self.hbar**2 * kx / self.effective_mass) * identity
            - self.alpha * sigma_y
        )

        dH_dky = (
            (self.hbar**2 * ky / self.effective_mass) * identity
            + self.alpha * sigma_x
        )

        return np.array([dH_dkx, dH_dky])