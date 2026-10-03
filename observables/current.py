import numpy as np

from .expectation import expectation_from_density_matrix


def charge_current(
    density_matrix,
    velocities,
    k_weight,
    charge=-1.0,
):
    """
    Charge-current density from a response density matrix.

    Parameters
    ----------
    density_matrix : ndarray, shape (Nk, nb, nb)
        Linear-response density matrix in the band basis.

    velocities : ndarray, shape (Nk, dim, nb, nb)
        Velocity operators in the band basis.

    k_weight : float
        k-space integration weight, including (2*pi)^(-d).

    charge : float
        Carrier charge. Default is -1 for electrons in units e=1.

    Returns
    -------
    current : ndarray, shape (dim,)
        Charge-current response

            j_alpha = q * sum_k w_k Tr[rho_k v_alpha,k].
    """

    dim = velocities.shape[1]

    current = np.zeros(dim, dtype=complex)

    for alpha in range(dim):
        current[alpha] = charge * expectation_from_density_matrix(
            density_matrix,
            velocities[:, alpha, :, :],
            k_weight,
        )

    return np.real_if_close(current)


def conductivity_from_current(
    current,
    electric_field,
    field_direction,
):
    """
    Conductivity response to an electric field applied along
    one specified direction.

        sigma_{alpha,beta} = j_alpha / E_beta

    where beta = field_direction.

    Parameters
    ----------
    current : ndarray, shape (dim,)
        Charge-current response.

    electric_field : ndarray, shape (dim,)
        Applied electric field.

    field_direction : int
        Direction beta of the applied electric field.

    Returns
    -------
    conductivity_column : ndarray, shape (dim,)
        sigma_{alpha,beta} for all current directions alpha.
    """

    E_beta = electric_field[field_direction]

    if np.isclose(E_beta, 0.0):
        raise ValueError(
            "Electric-field component along field_direction is zero."
        )

    return current / E_beta