import numpy as np
from scipy.optimize import brentq

from models.rashba import RashbaModel
from core.eigensystem import solve_on_mesh
from core.kspace import KMesh
from core.geometry import geometry_on_mesh, berry_curvature_on_mesh
from core.equilibrium import fermi_dirac
from driving.electric_field import diagonal_electric_drive
from disorder.scalar import ScalarDisorder
from disorder.collision import (
    scalar_collision_linear_operator,
    general_collision_linear_operator,
    diagonal_collision_from_coherence,
)
from response.coherence import (
    intrinsic_coherence,
    anomalous_driving,
    disorder_induced_coherence,
)
from observables.expectation import expectation_from_density_matrix
from observables.current import (
    charge_current,
    conductivity_from_current,
)
from response.density_matrix import assemble_density_matrix_response

from response.diagonal import (
    solve_leading_diagonal_response_iterative,
    solve_order_zero_diagonal_response,
)
from disorder.matrix import MatrixDisorder
# =============================================================================
# Configuration
# =============================================================================
MESH_SHAPE = (51, 51)          # use (61, 61) for final/convergence runs
K_BOUNDS = [(-1.5, 1.5), (-1.5, 1.5)]
RUN_N0_DIAGNOSTIC = True       # Eq. (49); kept separate from Eqs. (68)-(75)

MU = 0.5
ELECTRIC_FIELD = np.array([1.0, 0.0])
ETA_FERMI = 0.05
ETA_SCATTERING = 0.025
IMPURITY_DENSITY = 1.0
RTOL = 1e-13
DISORDER_TYPE = "scalar" # "scalar" or "matrix"
RUN_GENERAL_DISORDER_VALIDATION = True

model = RashbaModel(
    effective_mass=1.0,
    alpha=0.05,
    magnetization=0.005,
    hbar=1.0,
)

if DISORDER_TYPE == "scalar":

    disorder = ScalarDisorder(
        strength=1.0,
        n_bands=model.n_bands,
    )

elif DISORDER_TYPE == "matrix":

    U0 = 1.0
    Uz = 0.00

    sigma_z = np.array(
        [
            [1.0, 0.0],
            [0.0, -1.0],
        ],
        dtype=complex,
    )

    disorder_matrix = (
        U0 * np.eye(model.n_bands, dtype=complex)
        + Uz * sigma_z
    )

    disorder = MatrixDisorder(disorder_matrix)

else:
    raise ValueError(
        f"Unknown DISORDER_TYPE: {DISORDER_TYPE}"
    )


# =============================================================================
# Small analytic helpers used for validation
# =============================================================================
def analytic_rashba_curvature(k_points, alpha, magnetization):
    """Berry curvature for lower/upper magnetic Rashba bands."""
    k2 = np.sum(k_points**2, axis=1)
    magnitude = (
        magnetization * alpha**2
        / (2.0 * (magnetization**2 + alpha**2 * k2) ** 1.5)
    )
    # NumPy eigensystem ordering: band 0 = lower (-), band 1 = upper (+)
    return np.column_stack((magnitude, -magnitude))


def band_energy_radial(k, band, model):
    kinetic = model.hbar**2 * k**2 / (2.0 * model.effective_mass)
    splitting = np.sqrt(model.alpha**2 * k**2 + model.magnetization**2)
    if band == "lower":
        return kinetic - splitting
    if band == "upper":
        return kinetic + splitting
    raise ValueError("band must be 'lower' or 'upper'")


def find_fermi_wavevector(band, model, mu):
    return brentq(
        lambda k: band_energy_radial(k, band, model) - mu,
        0.0,
        5.0,
    )


def paper_ahe_predictions(model, kF_lower, kF_upper):
    """
    Magnetic Rashba AHE analytical results.

    Returns
    -------
    sigma_eq68 : float
        Literal Paper Eq. (68).

    sigma_eq69 : float
        Weak-M intrinsic result, Paper Eq. (69).

    sigma_eq74 : float
        Weak-M disorder-induced result, Paper Eq. (74).

    Notes
    -----
    With epsilon_± = kinetic ± lambda and alpha > 0, kF+ < kF-.
    Literal Eq. (68) therefore has the opposite sign from Eq. (69).
    Eqs. (69), (74), and the cancellation stated in Eq. (75)
    consistently give positive intrinsic and negative extrinsic
    Hall responses.
    """
    h = 2.0 * np.pi * model.hbar

    lambda_plus = np.sqrt(
        model.alpha**2 * kF_upper**2
        + model.magnetization**2
    )

    lambda_minus = np.sqrt(
        model.alpha**2 * kF_lower**2
        + model.magnetization**2
    )

    # Literal Paper Eq. (68)
    sigma_eq68 = -model.magnetization / (2.0 * h) * (
        1.0 / lambda_plus
        - 1.0 / lambda_minus
    )

    # Common kF approximation used in Eqs. (69) and (74)
    kF_average = 0.5 * (kF_lower + kF_upper)

    lambda_kF = np.sqrt(
        model.alpha**2 * kF_average**2
        + model.magnetization**2
    )

    common_weak_M_term = (
        1.0
        / (2.0 * h)
        * (
            2.0
            * model.alpha**2
            * model.effective_mass
            * model.magnetization
            / (model.hbar**2 * lambda_kF**2)
        )
    )

    # Paper Eq. (69): intrinsic
    sigma_eq69 = +common_weak_M_term

    # Paper Eq. (74): disorder-induced
    sigma_eq74 = -common_weak_M_term

    return sigma_eq68, sigma_eq69, sigma_eq74


def report_ahe(mesh_shape, intrinsic_vy, disorder_vy):
    intrinsic = float(np.real(intrinsic_vy))
    disorder_term = float(np.real(disorder_vy))
    residual = intrinsic + disorder_term
    relative_residual = abs(residual) / abs(intrinsic)

    print("\n" + "=" * 64)
    print("MAGNETIC RASHBA AHE — INTERBAND BENCHMARK")
    print("=" * 64)
    print(f"mesh                     : {mesh_shape[0]} x {mesh_shape[1]}")
    print(f"intrinsic vy             : {intrinsic:+.12e}")
    print(f"disorder-induced vy      : {disorder_term:+.12e}")
    print(f"interband residual vy    : {residual:+.12e}")
    print(f"relative residual        : {100.0 * relative_residual:.4f} %")
    print(f"cancellation             : {100.0 * (1.0-relative_residual):.4f} %")


# =============================================================================
# Main calculation
# =============================================================================
def main():
    # -------------------------------------------------------------------------
    # 1. k mesh, eigensystem, and geometry
    # -------------------------------------------------------------------------
    mesh = KMesh(bounds=K_BOUNDS, shape=MESH_SHAPE)
    k_points = mesh.generate()
    k_weight = mesh.integration_weight()

    energies, eigenvectors = solve_on_mesh(model, k_points)
    velocities, berry_connections = geometry_on_mesh(
        model, k_points, energies, eigenvectors
    )
    curvatures = berry_curvature_on_mesh(berry_connections)

    # High-value geometry regression test: numerical vs analytic Rashba curvature.
    analytic_curvatures = analytic_rashba_curvature(
        k_points, model.alpha, model.magnetization
    )
    curvature_error = np.max(np.abs(curvatures - analytic_curvatures))

    # -------------------------------------------------------------------------
    # 2. Leading diagonal response n_E^(-1)
    # -------------------------------------------------------------------------
    diagonal_drive = diagonal_electric_drive(
        energies,
        velocities,
        ELECTRIC_FIELD,
        MU,
        ETA_FERMI,
    )

    collision_operator = general_collision_linear_operator(
        disorder=disorder,
        k_points=k_points,
        energies=energies,
        eigenvectors=eigenvectors,
        eta=ETA_SCATTERING,
        k_weight=k_weight,
        impurity_density=IMPURITY_DENSITY,
        hbar=model.hbar,
    )

    if RUN_GENERAL_DISORDER_VALIDATION and DISORDER_TYPE == "scalar":

        print("\n" + "=" * 64)
        print("GENERAL DISORDER OPERATOR — SCALAR REGRESSION")
        print("=" * 64)

        # Old optimized scalar implementation
        scalar_operator = scalar_collision_linear_operator(
            disorder=disorder,
            energies=energies,
            eigenvectors=eigenvectors,
            eta=ETA_SCATTERING,
            k_weight=k_weight,
            impurity_density=IMPURITY_DENSITY,
            hbar=model.hbar,
        )

        # New general matrix-disorder implementation
        general_operator = general_collision_linear_operator(
            disorder=disorder,
            k_points=k_points,
            energies=energies,
            eigenvectors=eigenvectors,
            eta=ETA_SCATTERING,
            k_weight=k_weight,
            impurity_density=IMPURITY_DENSITY,
            hbar=model.hbar,
        )

        # Test both operators on the exact same arbitrary population vector.
        test_population = (
            np.arange(energies.size, dtype=float)
            .reshape(energies.shape)
        )

        test_population /= np.max(test_population)

        J_scalar = scalar_operator @ test_population.ravel()
        J_general = general_operator @ test_population.ravel()

        operator_difference = np.max(
            np.abs(J_scalar - J_general)
        )

        operator_scale = max(
            np.max(np.abs(J_scalar)),
            1e-30,
        )

        relative_operator_difference = (
            operator_difference / operator_scale
        )

        print(
            f"max |J_scalar - J_general| : "
            f"{operator_difference:.12e}"
        )
        print(
            f"relative difference         : "
            f"{relative_operator_difference:.12e}"
        )

    n_minus1 = solve_leading_diagonal_response_iterative(
        collision_operator=collision_operator,
        diagonal_drive=diagonal_drive,
        rtol=RTOL,
    )

    projected_drive = diagonal_drive.ravel() - np.mean(diagonal_drive)
    kinetic_residual = (
        collision_operator @ n_minus1.ravel() - projected_drive
    )

    # -------------------------------------------------------------------------
    # 3. Order-zero interband coherence
    # -------------------------------------------------------------------------
    occupations = fermi_dirac(energies, mu=MU, temperature=0.0)

    S_intrinsic = intrinsic_coherence(
        energies,
        occupations,
        berry_connections,
        ELECTRIC_FIELD,
    )

    anomalous_drive = anomalous_driving(
        model=model,
        disorder=disorder,
        k_points=k_points,
        energies=energies,
        eigenvectors=eigenvectors,
        n_minus1=n_minus1,
        eta=ETA_SCATTERING,
        k_weight=k_weight,
        impurity_density=IMPURITY_DENSITY,
    )

    S_disorder = disorder_induced_coherence(
        energies=energies,
        anomalous_drive=anomalous_drive,
        hbar=model.hbar,
    )

    # -------------------------------------------------------------------------
    # 4. Hall response from interband coherence — Paper Eqs. (68)-(75)
    # -------------------------------------------------------------------------
    vy_band = velocities[:, 1, :, :]

    intrinsic_vy = expectation_from_density_matrix(
        S_intrinsic, vy_band, k_weight
    )
    disorder_vy = expectation_from_density_matrix(
        S_disorder, vy_band, k_weight
    )

    # -------------------------------------------------------------------------
    # Physical charge-current / conductivity observables
    # -------------------------------------------------------------------------

    current_intrinsic = charge_current(
        density_matrix=S_intrinsic,
        velocities=velocities,
        k_weight=k_weight,
        charge=-1.0,
    )

    current_disorder = charge_current(
        density_matrix=S_disorder,
        velocities=velocities,
        k_weight=k_weight,
        charge=-1.0,
    )

    sigma_intrinsic = conductivity_from_current(
        current=current_intrinsic,
        electric_field=ELECTRIC_FIELD,
        field_direction=0,
    )

    sigma_disorder = conductivity_from_current(
        current=current_disorder,
        electric_field=ELECTRIC_FIELD,
        field_direction=0,
    )

    # Independent intrinsic check from anomalous velocity / Berry curvature.
    berry_vy = (
        -ELECTRIC_FIELD[0]
        / model.hbar
        * np.sum(k_weight * occupations * curvatures)
    )

    # -------------------------------------------------------------------------
    # 5. Compact validation report
    # -------------------------------------------------------------------------
    herm_intrinsic = np.max(
        np.abs(S_intrinsic - S_intrinsic.conj().transpose(0, 2, 1))
    )
    herm_disorder = np.max(
        np.abs(S_disorder - S_disorder.conj().transpose(0, 2, 1))
    )

    kF_lower = find_fermi_wavevector("lower", model, MU)
    kF_upper = find_fermi_wavevector("upper", model, MU)
    paper_eq68_sigma, paper_eq69_sigma, paper_eq74_sigma = (
        paper_ahe_predictions(
            model,
            kF_lower,
            kF_upper,
        )
    )

    print("\n" + "=" * 64)
    print("CORE NUMERICAL CHECKS")
    print("=" * 64)
    print(f"k-points                  : {len(k_points)}")
    print(f"integration weight        : {k_weight:.12e}")
    print(f"max Berry-curvature error : {curvature_error:.3e}")
    print(f"max n^-1 kinetic residual : {np.max(np.abs(kinetic_residual)):.3e}")
    print(f"S_int Hermiticity error   : {herm_intrinsic:.3e}")
    print(f"S_dis Hermiticity error   : {herm_disorder:.3e}")
    print(f"DM/Berry intrinsic diff   : {abs(intrinsic_vy-berry_vy):.3e}")
    print(
        f"current observable check   : "
        f"{abs(sigma_intrinsic[1] + intrinsic_vy/ELECTRIC_FIELD[0]):.3e}"
    )

    print("\n" + "=" * 64)
    print("MAGNETIC RASHBA — ANALYTIC PAPER BENCHMARK")
    print("=" * 64)
    print(f"kF lower (-)              : {kF_lower:.12f}")
    print(f"kF upper (+)              : {kF_upper:.12f}")
    print(
        f"M/(alpha*kF lower)       : "
        f"{model.magnetization/(model.alpha*kF_lower):.6f}"
    )
    print(
        f"M/(alpha*kF upper)       : "
        f"{model.magnetization/(model.alpha*kF_upper):.6f}"
    )

    print(
        f"alpha*kF lower / mu      : "
        f"{model.alpha*kF_lower/MU:.6f}"
    )

    print(
        f"alpha*kF upper / mu      : "
        f"{model.alpha*kF_upper/MU:.6f}"
    )

    # Observable code returns velocity.
    # For electron current, our convention gives sigma_yx = -vy / E_x.
    print(f"Paper Eq. (68), literal    : {paper_eq68_sigma:+.12e}")

    if DISORDER_TYPE == "scalar":

        intrinsic_sigma_num = np.real(sigma_intrinsic[1])
        disorder_sigma_num = np.real(sigma_disorder[1])

        intrinsic_relative_error = (
            abs(intrinsic_sigma_num - paper_eq69_sigma)
            / abs(paper_eq69_sigma)
        )

        disorder_relative_error = (
            abs(disorder_sigma_num - paper_eq74_sigma)
            / abs(paper_eq74_sigma)
        )

        interband_sigma_num = (
            intrinsic_sigma_num
            + disorder_sigma_num
        )

        paper_weak_M_total = (
            paper_eq69_sigma
            + paper_eq74_sigma
        )

        print(f"Paper Eq. (69), weak-M     : {paper_eq69_sigma:+.12e}")
        print(f"Our intrinsic sigma_xy     : {intrinsic_sigma_num:+.12e}")
        print(f"intrinsic relative error   : {intrinsic_relative_error:.3%}")

        print(f"Paper Eq. (74), weak-M     : {paper_eq74_sigma:+.12e}")
        print(f"Our extrinsic sigma_xy     : {disorder_sigma_num:+.12e}")
        print(f"extrinsic relative error   : {disorder_relative_error:.3%}")

        print(f"Paper Eq. (69) + Eq. (74)  : {paper_weak_M_total:+.12e}")
        print(f"Our interband sigma total  : {interband_sigma_num:+.12e}")

        report_ahe(
            MESH_SHAPE,
            intrinsic_vy,
            disorder_vy,
        )

    elif DISORDER_TYPE == "matrix":

        print("\n" + "=" * 64)
        print("GENERAL MATRIX-DISORDER RESPONSE")
        print("=" * 64)

        print(
            f"intrinsic sigma_xy        : "
            f"{np.real(sigma_intrinsic[1]):+.12e}"
        )

        print(
            f"disorder-induced sigma_xy : "
            f"{np.real(sigma_disorder[1]):+.12e}"
        )

        print(
            "Paper Eqs. (74)-(75)      : "
            "not applicable (scalar-disorder result)"
        )



    # -------------------------------------------------------------------------
    # 6. Order-zero diagonal response n_E^(0) — Paper Eq. (49)
    # -------------------------------------------------------------------------
    # Eq. (49):
    #     J_d[n_E^(0)] = -J_d[S_E^(0)]
    # with S_E^(0) = S_int^(0) + S_ext^(0).
    #
    # The paper's magnetic Rashba AHE benchmark, Eqs. (68)-(75), reports the
    # Hall response from S_int^(0) + S_ext^(0).  We therefore keep n_E^(0)
    # separate from that benchmark, while retaining it in the general response.
    n0 = None

    if RUN_N0_DIAGNOSTIC:
        S0 = S_intrinsic + S_disorder

        J_S0 = diagonal_collision_from_coherence(
            model=model,
            disorder=disorder,
            k_points=k_points,
            energies=energies,
            eigenvectors=eigenvectors,
            coherence=S0,
            eta=ETA_SCATTERING,
            k_weight=k_weight,
            impurity_density=IMPURITY_DENSITY,
        )

        n0 = solve_order_zero_diagonal_response(
            collision_operator=collision_operator,
            collision_from_coherence=J_S0,
            rtol=RTOL,
        )

        D_diag_0 = -J_S0.real
        projected_D_diag_0 = D_diag_0.ravel() - np.mean(D_diag_0)
        residual_n0 = (
            collision_operator @ n0.ravel() - projected_D_diag_0
        )

        vy_diagonal = np.diagonal(
            velocities[:, 1, :, :], axis1=1, axis2=2
        ).real
        vy_n0 = np.sum(k_weight * n0 * vy_diagonal)

        print("\n" + "=" * 64)
        print("GENERAL FRAMEWORK — EQ. (49) ORDER-ZERO RESPONSE")
        print("(reported separately from the Rashba Eqs. (68)-(75) benchmark)")
        print("=" * 64)
        print(f"max Im J_d[S^(0)]         : {np.max(np.abs(J_S0.imag)):.3e}")
        print(f"sum J_d[S^(0)]            : {np.sum(J_S0)}")
        print(f"sum n^(0)                 : {np.sum(n0):+.3e}")
        print(f"max n^(0) residual        : {np.max(np.abs(residual_n0)):.3e}")
        print(f"n^(0) transverse vy       : {vy_n0:+.12e}")
        print(f"n^(0) sigma_xy            : {-vy_n0/ELECTRIC_FIELD[0]:+.12e}")

    # -------------------------------------------------------------------------
    # 7. Assemble retained electric-field-induced density matrix
    # -------------------------------------------------------------------------
    rho_E = assemble_density_matrix_response(
        n_minus_1=n_minus1,
        S_intrinsic=S_intrinsic,
        S_disorder=S_disorder,
        n_zero=n0,
    )

    rho_hermiticity = np.max(
        np.abs(rho_E - rho_E.conj().transpose(0, 2, 1))
    )

    # -------------------------------------------------------------------------
    # Total physical current from the assembled response density matrix
    # -------------------------------------------------------------------------

    current_total = charge_current(
        density_matrix=rho_E,
        velocities=velocities,
        k_weight=k_weight,
        charge=-1.0,
    )

    sigma_total = conductivity_from_current(
        current=current_total,
        electric_field=ELECTRIC_FIELD,
        field_direction=0,
    )

    rho_n_minus_1 = np.zeros_like(rho_E, dtype=complex)

    idx = np.arange(model.n_bands)
    rho_n_minus_1[:, idx, idx] = n_minus1

    current_n_minus_1 = charge_current(
        density_matrix=rho_n_minus_1,
        velocities=velocities,
        k_weight=k_weight,
        charge=-1.0,
    )

    sigma_n_minus_1 = conductivity_from_current(
        current=current_n_minus_1,
        electric_field=ELECTRIC_FIELD,
        field_direction=0,
    )

    if n0 is not None:
        rho_n_zero = np.zeros_like(rho_E, dtype=complex)
        rho_n_zero[:, idx, idx] = n0

        current_n_zero = charge_current(
            density_matrix=rho_n_zero,
            velocities=velocities,
            k_weight=k_weight,
            charge=-1.0,
        )

        sigma_n_zero = conductivity_from_current(
            current=current_n_zero,
            electric_field=ELECTRIC_FIELD,
            field_direction=0,
        )

        print(
            f"n^(0) current check        : "
            f"{abs(sigma_n_zero[1] + vy_n0/ELECTRIC_FIELD[0]):.3e}"
        )
    else:
        sigma_n_zero = np.zeros_like(sigma_total)

    print("\n" + "=" * 64)
    print("ASSEMBLED RESPONSE DENSITY MATRIX")
    print("=" * 64)
    print(f"rho_E Hermiticity error   : {rho_hermiticity:.3e}")
    print(f"total current j_x          : {np.real(current_total[0]):+.12e}")
    print(f"total current j_y          : {np.real(current_total[1]):+.12e}")
    print(f"total sigma_xx             : {np.real(sigma_total[0]):+.12e}")
    print(f"total sigma_yx             : {np.real(sigma_total[1]):+.12e}")

    print("\n" + "=" * 64)
    print("HALL-RESPONSE DECOMPOSITION")
    print("=" * 64)

    print(
        f"diagonal n^(-1) sigma_yx  : "
        f"{np.real(sigma_n_minus_1[1]):+.12e}"
    )

    print(
        f"intrinsic S_int sigma_yx  : "
        f"{np.real(sigma_intrinsic[1]):+.12e}"
    )

    print(
        f"extrinsic S_ext sigma_yx  : "
        f"{np.real(sigma_disorder[1]):+.12e}"
    )

    print(
        f"diagonal n^(0) sigma_yx   : "
        f"{np.real(sigma_n_zero[1]):+.12e}"
    )

    print(
        f"interband total sigma_yx  : "
        f"{np.real(sigma_intrinsic[1] + sigma_disorder[1]):+.12e}"
    )

    print(
        f"full rho_E sigma_yx       : "
        f"{np.real(sigma_total[1]):+.12e}"
    )

    sigma_sum = (
        sigma_n_minus_1
        + sigma_n_zero
        + sigma_intrinsic
        + sigma_disorder
    )

    print(
        f"decomposition closure     : "
        f"{np.max(np.abs(sigma_total - sigma_sum)):.3e}"
    )


if __name__ == "__main__":
    main()