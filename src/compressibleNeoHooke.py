import numpy as np


def det33(matrix):

    a = matrix[0, 0]
    b = matrix[0, 1]
    c = matrix[0, 2]
    d = matrix[1, 0]
    e = matrix[1, 1]
    f = matrix[1, 2]
    g = matrix[2, 0]
    h = matrix[2, 1]
    i = matrix[2, 2]

    return a * e * i + b * f * g + c * d * h - c * e * g - b * d * i - a * f * h


def adjoint33(matrix):
    a11 = matrix[0, 0]
    a22 = matrix[1, 1]
    a33 = matrix[2, 2]
    a12 = matrix[0, 1]
    a13 = matrix[0, 2]
    a21 = matrix[1, 0]
    a23 = matrix[1, 2]
    a31 = matrix[2, 0]
    a32 = matrix[2, 1]

    a = a22 * a33 - a32 * a23
    b = a21 * a33 - a31 * a23
    c = a21 * a32 - a31 * a22

    d = a12 * a33 - a32 * a13
    e = a11 * a33 - a31 * a13
    f = a11 * a32 - a31 * a12

    g = a12 * a23 - a22 * a13
    h = a11 * a23 - a21 * a13
    i = a11 * a22 - a21 * a12

    cofactor = np.array([[a, -b, c], [-d, e, -f], [g, -h, i]])

    adjnt = cofactor.T

    return adjnt


def psi(C: np.ndarray, K: float, G: float) -> float:

    J = det33(C) ** (1.0 / 2)
    trC = C[0, 0] + C[1, 1] + C[2, 2]

    return K / 8.0 * (J - 1.0 / J) ** 2.0 + G / 2.0 * (trC * J ** (-2.0 / 3.0) - 3.0)


def dPsi_dC(C: np.ndarray, K: float, G: float) -> np.ndarray:

    detC = det33(C)
    J = detC ** (1.0 / 2)
    trC = C.trace()
    adjC = adjoint33(C)
    invC = adjC.T / detC

    dPsi_dJ = K / 4.0 * (J - 1.0 / J) * (1.0 + 1.0 / (J * J))
    dPsi_dJ -= G / 3.0 * trC * J ** (-5.0 / 3)

    dPsi_dtrC = G / 2.0 * J ** (-2.0 / 3)
    dtrCC_dC = np.identity(3)
    dJ_dC = invC * J * 0.5

    return dJ_dC * dPsi_dJ + dtrCC_dC * dPsi_dtrC


def d2Psi_dC_dC(C: np.ndarray, K: float, G: float) -> np.ndarray:

    detC = det33(C)
    J = detC ** (1.0 / 2)
    trC = C.trace()
    adjC = adjoint33(C)
    invC = adjC.T / detC

    dPsi_dJ = K / 4.0 * (J - 1.0 / J) * (1.0 + 1.0 / (J * J))
    dPsi_dJ -= G / 3.0 * trC * J ** (-5.0 / 3)

    dPsi_dtrC = G / 2.0 * J ** (-2.0 / 3)

    dtrC_dC = np.identity(3)

    dJ_dC = invC * J * 0.5

    d2Psi_dJ_dJ = K / 4.0 * (1.0 + 3.0 * J ** (-4))
    d2Psi_dJ_dJ += 5.0 * G / 9 * trC * J ** (-8.0 / 3)

    d2Psi_dJ_dtrC = -G / 3 * J ** (-5.0 / 3)

    d2J_dC_dC = 1.0 / 4 * J * np.einsum("ji,lk->ijkl", invC, invC)
    d2J_dC_dC -= 1.0 / 2 * J * np.einsum("jk,li->ijkl", invC, invC)

    return (
        d2Psi_dJ_dJ * np.einsum("ij,kl->ijkl", dJ_dC, dJ_dC)
        + dPsi_dJ * d2J_dC_dC
        + d2Psi_dJ_dtrC
        * (
            np.einsum("ij,kl->ijkl", dJ_dC, dtrC_dC)
            + np.einsum("ij,kl->ijkl", dtrC_dC, dJ_dC)
        )
    )


def computeSecondPiolaKirchhoffStress(C: np.ndarray, K: float, G: float) -> np.ndarray:
    return 2.0 * dPsi_dC(C, K, G)


def computeConsistentTangentFromPotential(
    C: np.ndarray, K: float, G: float
) -> np.ndarray:
    return 2.0 * d2Psi_dC_dC(C, K, G)
