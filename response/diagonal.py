import numpy as np


def solve_leading_diagonal_response(
    collision_matrix,
    diagonal_drive,
    rcond=1e-10
):
    drive_vector = diagonal_drive.ravel()

    collision_pinv = np.linalg.pinv(
        collision_matrix,
        rcond=rcond,
        hermitian=True
    )

    response_vector = (
        collision_pinv @ drive_vector
    )

    return response_vector.reshape(
        diagonal_drive.shape
    )

def solve_leading_diagonal_response_constrained(
    collision_matrix,
    diagonal_drive
):
    drive_vector = diagonal_drive.ravel()
    N = len(drive_vector)

    # Augmented system:
    #
    # [ K   1 ] [ n      ] = [ D ]
    # [ 1^T 0 ] [ lambda ]   [ 0 ]
    #
    # The final row imposes sum(n) = 0.

    augmented_matrix = np.zeros(
        (N + 1, N + 1),
        dtype=float
    )

    augmented_rhs = np.zeros(
        N + 1,
        dtype=float
    )

    augmented_matrix[:N, :N] = collision_matrix
    augmented_matrix[:N, N] = 1.0
    augmented_matrix[N, :N] = 1.0

    augmented_rhs[:N] = drive_vector

    solution = np.linalg.solve(
        augmented_matrix,
        augmented_rhs
    )

    response_vector = solution[:N]

    return response_vector.reshape(
        diagonal_drive.shape
    )

from scipy.sparse.linalg import minres


def solve_leading_diagonal_response_iterative(
    collision_operator,
    diagonal_drive,
    rtol=1e-10
):
    """
    Solve K n = D using MINRES.

    The collision operator has the zero mode K @ 1 = 0.
    We project the drive and final response onto the
    subspace orthogonal to that constant mode.
    """

    drive_vector = diagonal_drive.ravel().copy()

    # Remove any tiny numerical component along the zero mode
    drive_vector -= np.mean(drive_vector)

    response_vector, info = minres(
        collision_operator,
        drive_vector,
        rtol=rtol
    )

    if info != 0:
        raise RuntimeError(
            f"MINRES did not converge. info = {info}"
        )

    # Choose the physical solution with sum(n) = 0
    response_vector -= np.mean(response_vector)

    return response_vector.reshape(
        diagonal_drive.shape
    )

def solve_order_zero_diagonal_response(
    collision_operator,
    collision_from_coherence,
    rtol=1e-10,
):
    """
    Solve the order-zero diagonal kinetic equation

        J_d[n_E^(0)] = -J_d[S_E^(0)]

    corresponding to Eq. (49) of
    Culcer, Sekine & MacDonald, PRB 96, 035106 (2017).

    Parameters
    ----------
    collision_operator : LinearOperator
        Diagonal collision operator J_d.

    collision_from_coherence : ndarray, shape (Nk, nb)
        Diagonal projection J_d[S_E^(0)].

    rtol : float
        MINRES convergence tolerance.

    Returns
    -------
    n_zero : ndarray, shape (Nk, nb)
        Order-zero diagonal response n_E^(0).
    """

    drive = -np.real(collision_from_coherence)

    drive_vector = drive.ravel().copy()

    # Remove numerical component along the conserved zero mode
    drive_vector -= np.mean(drive_vector)

    response_vector, info = minres(
        collision_operator,
        drive_vector,
        rtol=rtol,
    )

    if info != 0:
        raise RuntimeError(
            f"MINRES did not converge for n_E^(0). info = {info}"
        )

    # Fix the arbitrary constant associated with the zero mode
    response_vector -= np.mean(response_vector)

    return response_vector.reshape(drive.shape)