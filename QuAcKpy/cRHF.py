#!/usr/bin/env python3
import numpy as np
import pandas as pd
import os


if __name__ == "__main__":
    # Inputs
    workdir = "../QuAcK/"
    eta = 0.01
    thresh = 0.00001
    maxSCF = 256
    nO = read_nO(workdir)
    maxDiis = 5

    # Read integrals
    T = read_matrix(f"{workdir}/int/Kin.dat")
    S = read_matrix(f"{workdir}/int/Ov.dat")
    V = read_matrix(f"{workdir}/int/Nuc.dat")
    ENuc = read_ENuc(f"{workdir}/int/ENuc.dat")
    nBas = np.shape(T)[0]
    nBasSq = nBas*nBas
    W = read_CAP_integrals(f"{workdir}/int/CAP.dat", nBas)
    ERI = read_2e_integrals(f"{workdir}/int/ERI.bin", nBas)
    X = get_X(S)
    W = -eta*W
    Hc = T + V + W*1j

    # Initialization
    F_diis = np.zeros((nBasSq, maxDiis))
    error_diis = np.zeros((nBasSq, maxDiis))
    rcond = 0

    # core guess
    _, c = diagonalize(X.T @ Hc @ X)
    c = X @ c
    P = 2*c[:, :nO]@c[:, :nO].T

    print('-' * 98)
    print(
        f"| {'#':<1} | {'E(RHF)':<36} | {'EJ(RHF)':<16} | {'EK(RHF)':<16} | {'Conv':<10} |")
    print('-' * 98)

    nSCF = 0
    Conv = 1
    n_diis = 0

    while(Conv > thresh and nSCF < maxSCF):
        nSCF += 1
        J = Hartree_matrix_AO_basis(P, ERI)
        K = exchange_matrix_AO_basis(P, ERI)
        F = Hc + J + 0.5*K
        err = F@P@S - S@P@F
        if nSCF > 1:
            Conv = np.max(np.abs(err))
        ET = np.trace(P@T)
        EV = np.trace(P@V)
        EJ = 0.5*np.trace(P@J)
        EK = 0.25*np.trace(P@K)
        ERHF = ET + EV + EJ + EK

      #  # DIIS
      #  n_diis = np.min([n_diis+1, maxDiis])
      #  rcond, n_diis, F = DIIS_extrapolation(
      #      rcond, n_diis, error_diis, F_diis, err, F)

        Fp = X.T @ F @ X
        eHF, c = diagonalize(Fp)
        c = X @ c
        P = 2*c[:, :nO]@c[:, :nO].T
        print(
            f"| {nSCF:3d} | {ERHF.real+ENuc:5.6f} + {ERHF.imag:5.6f}i | {EJ:5.6f} | {EK:5.6f} | {Conv:5.6f} |")
        print('-' * 98)
    print()
    print("RHF orbitals")
    print_matrix(eHF)
