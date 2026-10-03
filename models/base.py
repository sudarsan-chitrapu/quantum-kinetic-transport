"""
Base class for a Bloch Hamiltonian H(k).

Every physical model used by the transport framework
must inherit from this class.
"""

from abc import ABC, abstractmethod

import numpy as np


class BlochModel(ABC):
    def __init__(self, dimension: int, n_bands: int):
        self.dimension = dimension
        self.n_bands = n_bands

    @abstractmethod
    def H(self, k: np.ndarray) -> np.ndarray:
        """
        Return the Bloch Hamiltonian H(k).

        Parameters
        ----------
        k : np.ndarray
            Wave vector with shape (dimension,).

        Returns
        -------
        Hk : np.ndarray
            Complex Hermitian matrix with shape
            (n_bands, n_bands).
        """
        raise NotImplementedError

    @abstractmethod
    def dH_dk(self, k: np.ndarray) -> np.ndarray:
        """
        Return derivatives of H with respect to k.

        Returns
        -------
        dH : np.ndarray
            Shape:
            (dimension, n_bands, n_bands)

            dH[alpha] = dH / dk_alpha
        """
        raise NotImplementedError