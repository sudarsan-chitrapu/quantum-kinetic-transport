import numpy as np


def assemble_density_matrix_response(
    n_minus_1,
    S_intrinsic,
    S_disorder,
    n_zero=None,
):
    """
    Assemble the electric-field-induced density-matrix response.

    rho_E = n_E^(-1)
          + S_E,int^(0)
          + S_E,ext^(0)
          + n_E^(0)

    Parameters
    ----------
    n_minus_1 : ndarray, shape (Nk, nb)
        Leading band-diagonal response n_E^(-1).

    S_intrinsic : ndarray, shape (Nk, nb, nb)
        Intrinsic interband coherence S_E,int^(0).

    S_disorder : ndarray, shape (Nk, nb, nb)
        Disorder-induced interband coherence S_E,ext^(0).

    n_zero : ndarray, shape (Nk, nb), optional
        Order-zero diagonal correction n_E^(0).

    Returns
    -------
    rho_E : ndarray, shape (Nk, nb, nb)
        Total electric-field-induced density-matrix response
        to the retained order.
    """

    rho_E = np.array(
        S_intrinsic + S_disorder,
        dtype=complex,
        copy=True,
    )

    nb = rho_E.shape[1]
    idx = np.arange(nb)

    # Add leading diagonal response n_E^(-1)
    rho_E[:, idx, idx] += n_minus_1

    # Add order-zero diagonal correction when requested
    if n_zero is not None:
        rho_E[:, idx, idx] += n_zero

    return rho_E