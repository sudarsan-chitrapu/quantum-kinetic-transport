import numpy as np


def intrinsic_coherence(
    energies,
    occupations,
    berry_connections,
    electric_field,
    charge=1.0,
    degeneracy_tol=1e-10
):
    Nk, n_bands = energies.shape

    S_intrinsic = np.zeros(
        (Nk, n_bands, n_bands),
        dtype=complex
    )

    for i in range(Nk):

        # E · A_mn
        E_dot_A = np.einsum(
            "d,dmn->mn",
            electric_field,
            berry_connections[i]
        )

        for m in range(n_bands):
            for n in range(n_bands):

                if m == n:
                    continue

                energy_difference = (
                    energies[i, m]
                    - energies[i, n]
                )

                if abs(energy_difference) < degeneracy_tol:
                    S_intrinsic[i, m, n] = 0.0
                    continue

                occupation_difference = (
                    occupations[i, n]
                    - occupations[i, m]
                )

                S_intrinsic[i, m, n] = (
                    -charge
                    * occupation_difference
                    * E_dot_A[m, n]
                    / energy_difference
                )

    return S_intrinsic


def offdiagonal_collision_drive(
    model,
    disorder,
    k_points,
    energies,
    eigenvectors,
    n_minus1,
    eta,
    k_weight,
    impurity_density=1.0
):
    Nk, n_bands = energies.shape

    J_od = np.zeros(
        (Nk, n_bands, n_bands),
        dtype=complex
    )

    for i, k in enumerate(k_points):
        for j, kp in enumerate(k_points):

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

            for m in range(n_bands):
                for mp in range(n_bands):

                    if m == mp:
                        continue

                    for mpp in range(n_bands):

                        delta_1 = (
                            np.exp(
                                -(
                                    (energies[i, m]
                                     - energies[j, mpp])
                                    / eta
                                )**2
                            )
                            / (np.sqrt(np.pi) * eta)
                        )

                        delta_2 = (
                            np.exp(
                                -(
                                    (energies[i, mp]
                                     - energies[j, mpp])
                                    / eta
                                )**2
                            )
                            / (np.sqrt(np.pi) * eta)
                        )

                        term_1 = (
                            n_minus1[i, m]
                            - n_minus1[j, mpp]
                        ) * delta_1

                        term_2 = (
                            n_minus1[i, mp]
                            - n_minus1[j, mpp]
                        ) * delta_2

                        # IMPORTANT:
                        # D'_E = -J_od[n_E^(-1)].
                        #
                        # Eq. (48) of Culcer, Sekine & MacDonald PRB 96, 035106
                        # appears to miss this overall minus sign when compared with
                        # the definition preceding Eq. (47) and the explicit Rashba
                        # derivation Eqs. (60)-(61).
                        #
                        # This routine computes J_od[n_E^(-1)].
                        # anomalous_driving() applies the overall minus sign:
                        #
                        #     D'_E = -J_od[n_E^(-1)].

                        J_od[i, m, mp] += (
                            np.pi
                            * impurity_density
                            / model.hbar
                            * k_weight
                            * U_kkp[m, mpp]
                            * U_kpk[mpp, mp]
                            * (term_1 + term_2)
                        )

    return J_od

def anomalous_driving(
    model,
    disorder,
    k_points,
    energies,
    eigenvectors,
    n_minus1,
    eta,
    k_weight,
    impurity_density=1.0,
):
    """
    Disorder-induced anomalous driving term

        D'_E = -J_od[n_E^(-1)].

    Note:
    Eq. (48) of Culcer, Sekine & MacDonald,
    PRB 96, 035106 (2017), appears to contain an overall
    sign inconsistency. The definition preceding Eq. (47)
    and the explicit Rashba results Eqs. (60)-(61) and
    (72)-(75) require D'_E = -J_od.
    """

    J_od = offdiagonal_collision_drive(
        model=model,
        disorder=disorder,
        k_points=k_points,
        energies=energies,
        eigenvectors=eigenvectors,
        n_minus1=n_minus1,
        eta=eta,
        k_weight=k_weight,
        impurity_density=impurity_density,
    )

    return -J_od

def disorder_induced_coherence(
    energies,
    anomalous_drive,
    hbar=1.0,
    degeneracy_tol=1e-10
):
    Nk, n_bands = energies.shape

    S_disorder = np.zeros(
        (Nk, n_bands, n_bands),
        dtype=complex
    )

    for i in range(Nk):
        for m in range(n_bands):
            for n in range(n_bands):

                if m == n:
                    continue

                energy_difference = (
                    energies[i, m]
                    - energies[i, n]
                )

                if abs(energy_difference) < degeneracy_tol:
                    # Exclude points where the
                    # nondegenerate formula fails.
                    S_disorder[i, m, n] = 0.0
                    continue

                S_disorder[i, m, n] = (
                    -1j
                    * hbar
                    * anomalous_drive[i, m, n]
                    / energy_difference
                )

    return S_disorder