import numpy as np

from .base import DisorderModel


class ScalarDisorder(DisorderModel):

    def __init__(self, strength, n_bands):
        self.strength = strength
        self.n_bands = n_bands

    def matrix_element(self, k, kp):
        return (
            self.strength
            * np.eye(self.n_bands, dtype=complex)
        )