from abc import ABC, abstractmethod
import numpy as np


class DisorderModel(ABC):

    @abstractmethod
    def matrix_element(self, k, kp):
        """
        Disorder matrix element between
        orbital-basis Bloch states at k and k'.

        Returns an (n_bands, n_bands) matrix.
        """
        raise NotImplementedError