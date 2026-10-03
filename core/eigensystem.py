import numpy as np


def solve_eigensystem(model, k):

    Hk = model.H(k)

    energies, eigenvectors = np.linalg.eigh(Hk)

    return energies, eigenvectors

def solve_on_mesh(model, k_points):

    all_energies = []
    all_eigenvectors = []

    for k in k_points:

        energies, eigenvectors = solve_eigensystem(model, k)

        all_energies.append(energies)
        all_eigenvectors.append(eigenvectors)

    return (
        np.array(all_energies),
        np.array(all_eigenvectors)
    )