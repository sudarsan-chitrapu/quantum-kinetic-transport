import numpy as np
from scipy.sparse.linalg import LinearOperator
from .scattering import transition_rates


def diagonal_collision(
    populations,
    model,
    disorder,
    k_points,
    energies,
    eigenvectors,
    k_weight,
    eta,
    impurity_density=1.0
):
    Nk, n_bands = populations.shape

    collision = np.zeros_like(
        populations,
        dtype=float
    )

    for i, k in enumerate(k_points):
        for j, kp in enumerate(k_points):

            rates = transition_rates(
                disorder,
                k,
                kp,
                energies[i],
                energies[j],
                eigenvectors[i],
                eigenvectors[j],
                eta=eta,
                impurity_density=impurity_density,
                hbar=model.hbar
            )

            for m in range(n_bands):
                for n in range(n_bands):

                    collision[i, m] += (
                        k_weight
                        * rates[m, n]
                        * (
                            populations[i, m]
                            - populations[j, n]
                        )
                    )

    return collision

def build_diagonal_collision_matrix(
    model,
    disorder,
    k_points,
    energies,
    eigenvectors,
    k_weight,
    eta,
    impurity_density=1.0
):
    Nk, n_bands = energies.shape
    N = Nk * n_bands

    K = np.zeros((N, N), dtype=float)

    for i, k in enumerate(k_points):
        for j, kp in enumerate(k_points):

            rates = transition_rates(
                disorder,
                k,
                kp,
                energies[i],
                energies[j],
                eigenvectors[i],
                eigenvectors[j],
                eta=eta,
                impurity_density=impurity_density,
                hbar=model.hbar
            )

            for m in range(n_bands):
                for n in range(n_bands):

                    rate = k_weight * rates[m, n]

                    a = i * n_bands + m
                    b = j * n_bands + n

                    K[a, a] += rate
                    K[a, b] -= rate

    return K

def collision_matrix_from_rates(rates):

    Nk, n_bands, _, _ = rates.shape
    N = Nk * n_bands

    # Flatten (i,m,j,n) -> (a,b)
    W = rates.reshape(N, N)

    # K_ab = -W_ab
    K = -W.copy()

    # K_aa += sum_b W_ab
    total_out_rate = np.sum(W, axis=1)

    indices = np.arange(N)

    K[indices, indices] += total_out_rate

    return K

def collision_linear_operator(rates):
    """
    Matrix-free representation of the diagonal collision operator.

    (K n)_i = sum_j W_ij (n_i - n_j)

    rates has shape:
        (Nk, n_bands, Nk, n_bands)
    """

    Nk, n_bands, _, _ = rates.shape
    N = Nk * n_bands

    # Flatten W_(k,n,k',n') -> W_(i,j)
    W = rates.reshape(N, N)

    # Sum_j W_ij
    total_out_rate = np.sum(W, axis=1)

    def matvec(n):
        return (
            total_out_rate * n
            - W @ n
        )

    return LinearOperator(
        shape=(N, N),
        matvec=matvec,
        dtype=float
    )

def scalar_collision_linear_operator(
    disorder,
    energies,
    eigenvectors,
    eta,
    k_weight,
    impurity_density=1.0,
    hbar=1.0
):
    """
    Matrix-free collision operator for scalar disorder.

    Does NOT construct or store the full transition-rate tensor.
    """

    Nk, n_bands = energies.shape
    N = Nk * n_bands

    prefactor = (
        2.0 * np.pi
        * impurity_density
        * disorder.strength**2
        * k_weight
        / hbar
    )

    def matvec(n_vector):

        n = n_vector.reshape(Nk, n_bands)

        result = np.zeros_like(n)

        for i in range(Nk):

            # <u_n(k_i) | u_m(k_j)>
            overlaps = np.einsum(
                "an,jam->jnm",
                eigenvectors[i].conj(),
                eigenvectors
            )

            overlap_squared = np.abs(overlaps)**2

            # epsilon_n(k_i) - epsilon_m(k_j)
            energy_difference = (
                energies[i][None, :, None]
                - energies[:, None, :]
            )

            delta_energy = (
                np.exp(
                    -(energy_difference / eta)**2
                )
                / (np.sqrt(np.pi) * eta)
            )

            rates_i = (
                prefactor
                * overlap_squared
                * delta_energy
            )

            # n_n(k_i) - n_m(k_j)
            population_difference = (
                n[i][None, :, None]
                - n[:, None, :]
            )

            result[i] = np.sum(
                rates_i * population_difference,
                axis=(0, 2)
            )

        return result.ravel()

    return LinearOperator(
        shape=(N, N),
        matvec=matvec,
        dtype=float
    )

def diagonal_collision_from_coherence(
    model,
    disorder,
    k_points,
    energies,
    eigenvectors,
    coherence,
    eta,
    k_weight,
    impurity_density=1.0,
):
    """
    Diagonal projection of the energy-conserving Born
    collision integral acting on an off-diagonal density matrix S.

    Computes

        [J_delta(S)]_k^{mm}

    from Eq. (A1) of
    Culcer, Sekine & MacDonald, PRB 96, 035106 (2017).

    Parameters
    ----------
    coherence : ndarray, shape (Nk, nb, nb)
        Off-diagonal density matrix S_k.

    Returns
    -------
    result : ndarray, shape (Nk, nb)
        Diagonal elements [J_delta(S)]_k^{mm}.
    """

    Nk, nb = energies.shape

    result = np.zeros((Nk, nb), dtype=complex)

    prefactor = (
        np.pi
        * impurity_density
        * k_weight
        / model.hbar
    )

    def delta_gaussian(x):
        return (
            np.exp(-(x / eta) ** 2)
            / (np.sqrt(np.pi) * eta)
        )

    for i, k in enumerate(k_points):

        # S at the external momentum k
        S_k = coherence[i]

        for j, kp in enumerate(k_points):

            S_kp = coherence[j]

            # Disorder matrix elements in band basis
            #
            # U_kkp[a,b] = <a,k | U | b,k'>
            #
            U_kkp = (
                eigenvectors[i].conj().T
                @ disorder.matrix_element(k, kp)
                @ eigenvectors[j]
            )

            U_kpk = (
                eigenvectors[j].conj().T
                @ disorder.matrix_element(kp, k)
                @ eigenvectors[i]
            )

            # External/output band
            for m in range(nb):

                # --------------------------------------------------
                # Eq. (A1):
                #
                # We sum over the two internal band indices.
                #
                # The first two terms contain S(k).
                # The last two terms contain S(k').
                # --------------------------------------------------

                for a in range(nb):
                    for b in range(nb):

                        # S is supposed to be band off-diagonal.
                        # Explicitly excluding a == b makes that
                        # assumption clear and avoids accidental
                        # contamination from diagonal populations.
                        if a == b:
                            continue

                        # ==================================================
                        # TERMS 1 + 2
                        #
                        # Coherence evaluated at external momentum k.
                        # ==================================================

                        # Term 1:
                        #
                        # U_{k k'}^{m a}
                        # U_{k' k}^{a b}
                        # S_k^{b m}
                        #
                        delta_1 = delta_gaussian(
                            energies[i, b]
                            - energies[j, a]
                        )

                        term_1 = (
                            U_kkp[m, a]
                            * U_kpk[a, b]
                            * S_k[b, m]
                            * delta_1
                        )

                        # Term 2:
                        #
                        # S_k^{m b}
                        # U_{k k'}^{b a}
                        # U_{k' k}^{a m}
                        #
                        delta_2 = delta_gaussian(
                            energies[i, b]
                            - energies[j, a]
                        )

                        term_2 = (
                            S_k[m, b]
                            * U_kkp[b, a]
                            * U_kpk[a, m]
                            * delta_2
                        )

                        # ==================================================
                        # TERMS 3 + 4
                        #
                        # Coherence evaluated at scattered momentum k'.
                        # ==================================================

                        # Term 3:
                        #
                        # - U_{k k'}^{m a}
                        #   S_{k'}^{a b}
                        #   U_{k' k}^{b m}
                        #
                        delta_3 = delta_gaussian(
                            energies[i, m]
                            - energies[j, b]
                        )

                        term_3 = (
                            -U_kkp[m, a]
                            * S_kp[a, b]
                            * U_kpk[b, m]
                            * delta_3
                        )

                        # Term 4:
                        #
                        # - U_{k k'}^{m a}
                        #   S_{k'}^{a b}
                        #   U_{k' k}^{b m}
                        #
                        # but with the second energy-conservation
                        # condition from Eq. (A1).
                        #
                        delta_4 = delta_gaussian(
                            energies[i, m]
                            - energies[j, a]
                        )

                        term_4 = (
                            -U_kkp[m, a]
                            * S_kp[a, b]
                            * U_kpk[b, m]
                            * delta_4
                        )

                        result[i, m] += prefactor * (
                            term_1
                            + term_2
                            + term_3
                            + term_4
                        )

    return result

def general_collision_linear_operator(
    disorder,
    k_points,
    energies,
    eigenvectors,
    eta,
    k_weight,
    impurity_density=1.0,
    hbar=1.0,
):
    """
    Matrix-free diagonal collision operator for a general
    matrix-valued disorder potential U(k, k').

    Implements

        (J_d n)_{k,m}
            = sum_{k',n} W_{k m, k' n}
              [n_{k,m} - n_{k',n}]

    with

        W_{k m, k' n}
            = (2*pi*n_i/hbar)
              |<u_{m,k}|U(k,k')|u_{n,k'}>|^2
              delta(eps_{m,k} - eps_{n,k'}).

    Unlike scalar_collision_linear_operator(), this does not
    assume U(k,k') = U0 * I.
    """

    Nk, n_bands = energies.shape
    N = Nk * n_bands

    prefactor = (
        2.0
        * np.pi
        * impurity_density
        * k_weight
        / hbar
    )

    def matvec(n_vector):

        n = n_vector.reshape(Nk, n_bands)

        result = np.zeros_like(n, dtype=float)

        for i, k in enumerate(k_points):

            rates_i = np.zeros(
                (Nk, n_bands, n_bands),
                dtype=float,
            )

            for j, kp in enumerate(k_points):

                # Disorder matrix element in orbital/spin basis
                U_orbital = disorder.matrix_element(k, kp)

                # Transform to band basis:
                #
                # U_band[m,n]
                # = <u_{m,k}|U(k,k')|u_{n,k'}>
                U_band = (
                    eigenvectors[i].conj().T
                    @ U_orbital
                    @ eigenvectors[j]
                )

                matrix_element_squared = np.abs(U_band) ** 2

                energy_difference = (
                    energies[i][:, None]
                    - energies[j][None, :]
                )

                delta_energy = (
                    np.exp(
                        -(energy_difference / eta) ** 2
                    )
                    / (np.sqrt(np.pi) * eta)
                )

                rates_i[j] = (
                    prefactor
                    * matrix_element_squared
                    * delta_energy
                )

            population_difference = (
                n[i][None, :, None]
                - n[:, None, :]
            )

            result[i] = np.sum(
                rates_i * population_difference,
                axis=(0, 2),
            )

        return result.ravel()

    return LinearOperator(
        shape=(N, N),
        matvec=matvec,
        dtype=float,
    )