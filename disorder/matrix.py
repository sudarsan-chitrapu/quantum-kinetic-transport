import numpy as np

from .base import DisorderModel


class MatrixDisorder(DisorderModel):
    """
    Momentum-independent matrix-valued disorder potential.

    U(k, k') = U_matrix

    Useful for testing spin/orbital-dependent disorder.
    """

    def __init__(self, matrix):
        matrix = np.asarray(matrix, dtype=complex)

        if matrix.ndim != 2:
            raise ValueError("Disorder matrix must be two-dimensional.")

        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Disorder matrix must be square.")

        # Static disorder potential should be Hermitian.
        if not np.allclose(matrix, matrix.conj().T):
            raise ValueError("Disorder matrix must be Hermitian.")

        self.matrix = matrix
        self.n_bands = matrix.shape[0]

    def matrix_element(self, k, kp):
        return self.matrix