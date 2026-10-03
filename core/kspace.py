import numpy as np


class KMesh:

    def __init__(self, bounds, shape):

        self.bounds = bounds
        self.shape = shape

        self.dimension = len(bounds)

    def generate(self):

        axes = []

        for (k_min, k_max), n_points in zip(self.bounds, self.shape):
            axis = np.linspace(k_min, k_max, n_points)
            axes.append(axis)

        grids = np.meshgrid(*axes, indexing="ij")

        points = np.stack(
            [grid.ravel() for grid in grids],
            axis=-1
        )

        return points

    def integration_weight(self):
        weight = 1.0

        for (k_min, k_max), n_points in zip(
            self.bounds,
            self.shape
        ):
            if n_points < 2:
                raise ValueError(
                    "Each dimension needs at least 2 k-points."
                )

            dk = (k_max - k_min) / (n_points - 1)

            weight *= dk

        weight /= (2.0 * np.pi)**self.dimension

        return weight