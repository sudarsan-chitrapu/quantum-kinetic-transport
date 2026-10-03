import numpy as np


def expectation_from_density_matrix(
    density_matrix,
    operator_band,
    k_weight
):
    """
    Compute sum_k w_k Tr[rho(k) O(k)].
    """

    expectation = 0.0 + 0.0j

    for i in range(len(density_matrix)):
        expectation += (
            k_weight
            * np.trace(
                density_matrix[i]
                @ operator_band[i]
            )
        )

    return expectation