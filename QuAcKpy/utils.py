import numpy as np
import pandas as pd
import os


def generate_cap_file(filename, size, width, seed=42):
    """
    Generates a file with random CAP integral values in the format:
    mu nu wx wy wz
    """
    np.random.seed(seed)  # For reproducibility
    with open(filename, 'w') as f:
        for mu in range(1, size + 1):
            for nu in range(mu, size + 1):  # Only upper triangle to avoid duplicate entries
                # Generate three random values
                wx, wy, wz = np.random.rand(3)*width
                f.write(f"{mu} {nu} {wx:.6f} {wy:.6f} {wz:.6f}\n")


def print_matrix(matrix):
    df = pd.DataFrame(matrix)
    print(df)


def read_nO(working_dir):
    file_path = os.path.join(working_dir, "input/molecule")
    print(file_path)
    with open(file_path, "r") as f:
        next(f)  # Skip the first line
        line = f.readline().split()
        nO = max(int(line[1]), int(line[2]))
    return nO


def read_ENuc(file_path):
    # Path to the ENuc.dat file
    with open(file_path, 'r') as f:
        # Read the nuclear repulsion energy from the first line
        ENuc = float(f.readline().strip())
    return ENuc


def read_matrix(filename):
    # Read the data and determine matrix size
    entries = []
    max_index = 0

    with open(filename, 'r') as f:
        for line in f:
            i, j, value = line.split()
            i, j = int(i) - 1, int(j) - 1  # Convert to zero-based index
            entries.append((i, j, float(value)))
            # Find max index to determine size
            max_index = max(max_index, i, j)

    # Initialize square matrix with zeros
    matrix = np.zeros((max_index + 1, max_index + 1))

    # Fill the matrix
    for i, j, value in entries:
        matrix[i, j] = value
        if i != j:  # Assuming the matrix is symmetric, fill the transpose element
            matrix[j, i] = value

    return matrix


def read_CAP_integrals(filename, size):
    """
    Reads the file and constructs the symmetric matrix W.
    """
    W = np.zeros((size, size))
    with open(filename, 'r') as f:
        for line in f:
            mu, nu, wx, wy, wz = line.split()
            mu, nu = int(mu) - 1, int(nu) - 1  # Convert to zero-based index
            value = float(wx) + float(wy) + float(wz)
            W[mu, nu] = value
            W[nu, mu] = value  # Enforce symmetry
    return W


def read_2e_integrals(file_path, nBas):
    # Read the binary file and reshape the data into a 4D array
    try:
        G = np.fromfile(file_path, dtype=np.float64).reshape(
            (nBas, nBas, nBas, nBas))
    except FileNotFoundError:
        print(f"Error opening file: {file_path}")
        raise
    return G


def get_X(S):
    """
       Computes matrix X for orthogonalization. Attention O has to be hermitian.
    """
    vals, U = np.linalg.eigh(S)
    # Sort the eigenvalues and eigenvectors
    vals = 1/np.sqrt(vals)
    return U@np.diag(vals)


def sort_eigenpairs(eigenvalues, eigenvectors):
    # Get the sorting order based on the real part of the eigenvalues
    order = np.argsort(eigenvalues.real)

    # Sort eigenvalues and eigenvectors
    sorted_eigenvalues = eigenvalues[order]
    sorted_eigenvectors = eigenvectors[:, order]
    return sorted_eigenvalues, sorted_eigenvectors


def diagonalize_gram_schmidt(M):
    # Diagonalize the matrix
    vals, vecs = np.linalg.eig(M)
    # Sort the eigenvalues and eigenvectors
    vals, vecs = sort_eigenpairs(vals, vecs)
    # Orthonormalize them wrt cTc inner product
    vecs = gram_schmidt(vecs)
    return vals, vecs


def diagonalize(M):
    # Diagonalize the matrix
    vals, vecs = np.linalg.eig(M)
    # Sort the eigenvalues and eigenvectors
    vals, vecs = sort_eigenpairs(vals, vecs)
    # Orthonormalize them wrt cTc inner product
    vecs = orthonormalize(vecs)
    return vals, vecs


def orthonormalize(vecs):
    # Orthonormalize them wrt cTc inner product
    R = vecs.T@vecs
    L = cholesky_decomposition(R)
    Linv = np.linalg.inv(L)
    vecs = vecs@Linv.T
    return vecs


def Hartree_matrix_AO_basis(nBas, P, ERI):
    # Initialize Hartree matrix with zeros (complex type)
    J = np.zeros((nBas, nBas), dtype=np.complex128)

    # Compute Hartree matrix
    for si in range(nBas):
        for nu in range(nBas):
            for la in range(nBas):
                for mu in range(nBas):
                    J[mu, nu] += P[la, si] * ERI[mu, la, nu, si]

    return J


def exchange_matrix_AO_basis(nBas, P, ERI):
    # Initialize exchange matrix with zeros
    K = np.zeros((nBas, nBas), dtype=np.complex128)

    # Compute exchange matrix
    for nu in range(nBas):
        for si in range(nBas):
            for la in range(nBas):
                for mu in range(nBas):
                    K[mu, nu] -= P[la, si] * ERI[mu, la, si, nu]
    return K


def gram_schmidt(vectors):
    """
    Orthonormalize a set of vectors with respect to the scalar product c^T c.
    """
    orthonormal_basis = []
    for v in vectors.T:  # Iterate over column vectors
        for u in orthonormal_basis:
            v -= (u.T @ v) * u  # Projection with respect to c^T c
        norm = np.sqrt(v.T @ v)  # Norm with respect to c^T c
        if norm > 1e-10:
            orthonormal_basis.append(v / norm)
        else:
            raise Exception("Norm of eigenvector < 1e-10")
    return np.column_stack(orthonormal_basis)


def DIIS_extrapolation(rcond, n_diis, error, e, error_in, e_inout):
    """
    Perform DIIS extrapolation.

    """

    # Update DIIS history by prepending new error and solution vectors
    error = np.column_stack((error_in, error[:, :-1]))  # Shift history
    e = np.column_stack((e_inout, e[:, :-1]))          # Shift history

    # Build A matrix
    A = np.zeros((n_diis + 1, n_diis + 1), dtype=np.complex128)
    print(np.shape(error))
    A[:n_diis, :n_diis] = error@error.T
    A[:n_diis, n_diis] = -1.0
    A[n_diis, :n_diis] = -1.0
    A[n_diis, n_diis] = 0.0

    # Build b vector
    b = np.zeros(n_diis + 1, dtype=np.complex128)
    b[n_diis] = -1.0

    # Solve the linear system A * w = b
    try:
        w = np.linalg.solve(A, b)
        rcond = np.linalg.cond(A)
    except np.linalg.LinAlgError:
        raise ValueError("DIIS linear system is singular or ill-conditioned.")

    # Extrapolate new solution
    e_inout[:] = w[:n_diis]@e[:, :n_diis].T

    return rcond, n_diis, e_inout


def cholesky_decomposition(A):
    """
    Performs Cholesky-Decomposition wrt the c product. Returns L such that A = LTL
    """

    L = np.zeros_like(A)
    n = np.shape(L)[0]
    for i in range(n):
        for j in range(i + 1):
            s = A[i, j]

            for k in range(j):
                s -= L[i, k] * L[j, k]

            if i > j:  # Off-diagonal elements
                L[i, j] = s / L[j, j]
            else:  # Diagonal elements
                L[i, i] = s**0.5

    return L
