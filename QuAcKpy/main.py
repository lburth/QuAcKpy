from .utils import *
import os
import sys
import argparse
import pyscf
from pyscf import gto
import numpy as np
import time
import subprocess


def main():
    workdir = "./"

    # Create the argument parser object and gives a description of the script
    parser = argparse.ArgumentParser(
        description='This script is the main script of CAPybara, it is used to run the calculation.\n')
    # Create the argument parser object and gives a description of the script
    parser.add_argument('-b', '--basis', type=str, required=True,
                        help='Name of the file containing the basis set in the $QUACK_ROOT/basis/ directory')
    parser.add_argument('--bohr', default='Angstrom', action='store_const', const='Bohr',
                        help='By default QuAcK assumes that the xyz files are in Angstrom. Add this argument if your xyz file is in Bohr.')
    parser.add_argument('-c', '--charge', type=int, default=0,
                        help='Total charge of the molecule. Specify negative charges with "m" instead of the minus sign, for example m1 instead of -1. Default is 0')
    parser.add_argument('--cartesian', default=False, action='store_true',
                        help='Add this option if you want to use cartesian basis functions.')
    parser.add_argument('--print_2e', default=True, action='store_true',
                        help='If True, print 2e-integrals to disk.')
    parser.add_argument('--formatted_2e', default=False, action='store_true',
                        help='Add this option if you want to print formatted 2e-integrals.')
    parser.add_argument('--mmap_2e', default=False, action='store_true',
                        help='If True, avoid using DRAM when generating 2e-integrals.')
    parser.add_argument('--aosym_2e', default=False, action='store_true',
                        help='If True, use 8-fold symmetry 2e-integrals.')
    parser.add_argument('-fc', '--frozen_core', type=bool,
                        default=False, help='Freeze core MOs. Default is false')
    parser.add_argument('-m', '--multiplicity', type=int, default=1,
                        help='Spin multiplicity. Default is 1 therefore singlet')
    parser.add_argument('--working_dir', type=str, default=workdir,
                        help='Set a working directory to run the calculation.')
    parser.add_argument('-x', '--xyz', type=str, required=True,
                        help='Name of the file containing the nuclear coordinates in xyz format in the $QUACK_ROOT/mol/ directory without the .xyz extension')

    # Parse the arguments
    args = parser.parse_args()
    input_basis = args.basis
    unit = args.bohr
    charge = args.charge
    frozen_core = args.frozen_core
    multiplicity = args.multiplicity
    xyz = args.xyz + '.xyz'
    cartesian = args.cartesian
    print_2e = args.print_2e
    formatted_2e = args.formatted_2e
    mmap_2e = args.mmap_2e
    aosym_2e = args.aosym_2e
    working_dir = args.working_dir

    # Read molecule
    f = open(working_dir+'/mol/'+xyz, 'r')
    lines = f.read().splitlines()
    nbAt = int(lines.pop(0))
    lines.pop(0)
    list_pos_atom = []
    for line in lines:
        tmp = line.split()
        atom = tmp[0]
        pos = (float(tmp[1]), float(tmp[2]), float(tmp[3]))
        list_pos_atom.append([atom, pos])
    f.close()

    # Definition of the molecule
    mol = gto.M(
        atom=list_pos_atom,
        basis=input_basis,
        charge=charge,
        spin=multiplicity - 1
        #    symmetry = True  # Enable symmetry
    )

    # Fix the unit for the lengths
    mol.unit = unit
    #
    mol.cart = cartesian

    # Update mol object
    mol.build()

    # Accessing number of electrons
    nelec = mol.nelec  # Access the number of electrons
    nalpha = nelec[0]
    nbeta = nelec[1]

    subprocess.call(['mkdir', '-p', working_dir+'/input'])
    f = open(working_dir+'/input/molecule', 'w')
    f.write('# nAt nEla nElb nCore nRyd\n')
    f.write(str(mol.natm)+' '+str(nalpha)+' ' +
            str(nbeta)+' '+str(0)+' '+str(0)+'\n')
    f.write('# Znuc x  y  z\n')
    for i in range(len(list_pos_atom)):
        f.write(list_pos_atom[i][0]+' '+str(list_pos_atom[i][1][0])+' ' +
                str(list_pos_atom[i][1][1])+' '+str(list_pos_atom[i][1][2])+'\n')
    f.close()

    # Compute nuclear energy and put it in a file
    subprocess.call(['mkdir', '-p', working_dir+'/int'])
    subprocess.call(['rm', '-f', working_dir + '/int/ENuc.dat'])
    f = open(working_dir+'/int/ENuc.dat', 'w')
    f.write(str(mol.energy_nuc()))
    f.write(' ')
    f.close()

    # Compute 1e integrals
    ovlp = mol.intor('int1e_ovlp')  # Overlap matrix elements
    v1e = mol.intor('int1e_nuc')  # Nuclear repulsion matrix elements
    t1e = mol.intor('int1e_kin')  # Kinetic energy matrix elements
    dipole = mol.intor('int1e_r')  # Matrix elements of the x, y, z operators
    x, y, z = dipole[0], dipole[1], dipole[2]

    norb = len(ovlp)  # nBAS_AOs
    subprocess.call(['rm', '-f', working_dir + '/int/nBas.dat'])
    f = open(working_dir+'/int/nBas.dat', 'w')
    f.write(" {} ".format(str(norb)))
    f.close()

    def write_matrix_to_file(matrix, size, file, cutoff=1e-15):
        f = open(file, 'w')
        for i in range(size):
            for j in range(i, size):
                if abs(matrix[i][j]) > cutoff:
                    f.write(str(i+1)+' '+str(j+1)+' ' +
                            "{:.16E}".format(matrix[i][j]))
                    f.write('\n')
        f.close()

    # Write all 1 electron quantities in files
    # Ov,Nuc,Kin,x,y,z
    subprocess.call(['rm', '-f', working_dir + '/int/Ov.dat'])
    write_matrix_to_file(ovlp, norb, working_dir+'/int/Ov.dat')
    subprocess.call(['rm', '-f', working_dir + '/int/Nuc.dat'])
    write_matrix_to_file(v1e, norb, working_dir+'/int/Nuc.dat')
    subprocess.call(['rm', '-f', working_dir + '/int/Kin.dat'])
    write_matrix_to_file(t1e, norb, working_dir+'/int/Kin.dat')
    subprocess.call(['rm', '-f', working_dir + '/int/x.dat'])
    write_matrix_to_file(x, norb, working_dir+'/int/x.dat')
    subprocess.call(['rm', '-f', working_dir + '/int/y.dat'])
    write_matrix_to_file(y, norb, working_dir+'/int/y.dat')
    subprocess.call(['rm', '-f', working_dir + '/int/z.dat'])
    write_matrix_to_file(z, norb, working_dir+'/int/z.dat')

    def write_tensor_to_file(tensor, size, file_name, cutoff=1e-15):
        f = open(file_name, 'w')
        for i in range(size):
            for j in range(i, size):
                for k in range(i, size):
                    for l in range(j, size):
                        if abs(tensor[i][k][j][l]) > cutoff:
                            f.write(str(i+1)+' '+str(j+1)+' '+str(k+1)+' ' +
                                    str(l+1)+' '+"{:.16E}".format(tensor[i][k][j][l]))
                            f.write('\n')
        f.close()

    if print_2e:
        # Write two-electron integrals to HD
        ti_2e = time.time()

        if formatted_2e:
            output_file_path = working_dir + '/int/ERI.dat'
            subprocess.call(['rm', '-f', output_file_path])
            eri_ao = mol.intor('int2e')
            write_tensor_to_file(eri_ao, norb, output_file_path)

        if aosym_2e:
            output_file_path = working_dir + '/int/ERI_chem.bin'
            subprocess.call(['rm', '-f', output_file_path])
            eri_ao = mol.intor('int2e', aosym='s8')
            f = open(output_file_path, 'w')
            eri_ao.tofile(output_file_path)
            f.close()
        else:
            output_file_path = working_dir + '/int/ERI.bin'
            subprocess.call(['rm', '-f', output_file_path])
            if(mmap_2e):
                # avoid using DRAM
                eri_shape = (norb, norb, norb, norb)
                eri_mmap = np.memmap(
                    output_file_path, dtype='float64', mode='w+', shape=eri_shape)
                mol.intor('int2e', out=eri_mmap)
                for i in range(norb):
                    eri_mmap[i, :, :, :] = eri_mmap[i,
                                                    :, :, :].transpose(1, 0, 2)
                eri_mmap.flush()
                del eri_mmap
            else:
                eri_ao = mol.intor('int2e').transpose(
                    0, 2, 1, 3)  # chem -> phys
                f = open(output_file_path, 'w')
                eri_ao.tofile(output_file_path)
                f.close()

        te_2e = time.time()
        print(
            "Wall time for writing 2e-integrals to disk: {:.3f} seconds".format(te_2e - ti_2e))
        sys.stdout.flush()

    # Put this at another point at some time
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
    subprocess.call(['rm', '-f', working_dir + '/int/CAP.dat'])
    # Remove this at some point by calculating actually the integrals
    generate_cap_file(f"{workdir}/int/CAP.dat", nBas, 1)
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
        J = Hartree_matrix_AO_basis(nBas, P, ERI)
        K = exchange_matrix_AO_basis(nBas, P, ERI)
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
