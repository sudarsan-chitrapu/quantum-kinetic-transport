# Quantum Kinetic Theory of Electronic Transport in Multiband Systems
A Computational Framework for Interband Coherence, Disorder, and Electronic Response

1. Introduction

Understanding electronic transport in multiband quantum systems requires going beyond the semiclassical picture of independent carriers moving within individual energy bands. In systems with spin-orbit coupling, broken symmetries, or multiple coupled bands, an applied electric field can generate interband quantum coherence in addition to changes in band populations. This coherence underlies a range of transport phenomena, including anomalous and spin Hall effects, spin-orbit torques, and other Berry-phase-driven responses.

Disorder further complicates this picture. Scattering not only redistributes carriers within and between bands, but can also generate additional interband coherence. A complete transport theory must therefore describe the interplay between band structure, Berry geometry, electric-field driving, and disorder scattering within a common framework.

This repository develops a modular computational implementation of the density-matrix quantum kinetic approach to multiband electronic transport. The starting point is the formalism developed by Culcer, Sekine, and MacDonald for separating the electric-field response into band-diagonal and band-off-diagonal components and systematically organizing them according to their dependence on disorder. The project is being developed progressively across a sequence of theoretical works, with the longer-term goal of building a reusable framework that can be applied beyond analytically tractable model Hamiltonians.

The current implementation includes the leading band-diagonal response, intrinsic and disorder-induced interband coherence, the order-zero diagonal correction, charge-current observables, and elastic Born scattering for both scalar and matrix-valued disorder potentials. The framework is presently validated using the magnetic Rashba model and its anomalous Hall response, where the intrinsic and disorder-induced contributions provide a particularly useful test of the implementation.

Rather than hard-coding the transport calculation for a specific Hamiltonian, the code separates the electronic model from the transport machinery. A model supplies its Bloch Hamiltonian (H(\mathbf{k})) and momentum derivatives, while the remaining modules construct the eigensystem, geometric quantities, collision operators, density-matrix response, and observables. This architecture is intended to support progressively more realistic multiband models and, ultimately, electronic structures derived from first-principles and Wannier-based calculations.

Current status: Core linear-response quantum kinetic framework implemented through order (n_i^0) for nondegenerate multiband systems with energy-conserving Born scattering. Scalar and matrix-valued disorder are supported, with the present implementation benchmarked against the magnetic Rashba anomalous Hall problem.
