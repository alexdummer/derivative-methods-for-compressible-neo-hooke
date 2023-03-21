import numpy as np
from .tools import computePerturbationScalingForArray
from .compressibleNeoHooke import psi
from .compressibleNeoHooke import dPsi_dC as dPsi_dC_analytical

def dPsi_dC(
    C: np.ndarray, K: float, G: float, h: float = np.finfo(float).eps ** (1.0 / 3)
) -> np.ndarray:

    dPsi_dC_ = np.zeros_like(C)

    h_ = computePerturbationScalingForArray(C) * h

    for i in range(3):
        for j in range(3):
            C_right = np.copy(C)
            C_right_right = np.copy(C)
            C_right[i, j] += h_[i, j]
            C_right_right[i, j] += 2.0 * h_[i, j]
            dPsi_dC_[i, j] = (
                4.0 * psi(C_right, K, G) - psi(C_right_right, K, G) - 3.0 * psi(C, K, G)
            ) / (2.0 * h_[i, j])

    return dPsi_dC_


def dS_dC(
    C: np.ndarray, K: float, G: float, h: float = np.sqrt(np.finfo(float).eps)
) -> np.ndarray:

    dS_dC_ = np.zeros((3, 3, 3, 3))

    dPsi_dC_analytical_ = dPsi_dC_analytical(C, K, G)

    h_ = computePerturbationScalingForArray(C) * h

    for k in range(3):
        for l in range(3):
            C_right = np.copy(C)
            C_right_right = np.copy(C)
            C_right[k, l] += h_[k, l]
            C_right_right[k, l] += 2.0 * h_[k, l]
            dS_dC_[:, :, k, l] = (
                4.0 * dPsi_dC_analytical(C_right, K, G)
                - dPsi_dC_analytical(C_right_right, K, G)
                - 3.0 * dPsi_dC_analytical_
            ) / (2.0 * h_[k, l])

    return 2.0 * dS_dC_


def d2Psi_dC_dC(
    C: np.ndarray, K: float, G: float, h: float = np.finfo(float).eps ** (1.0 / 4)
) -> np.ndarray:

    dS_dC_1 = np.zeros((3, 3, 3, 3))

    h_ = computePerturbationScalingForArray(C) * h

    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    C1 = np.copy(C)
                    C2 = np.copy(C)
                    C3 = np.copy(C)

                    C1[i, j] += h_[i, j]
                    C1[k, l] += h_[k, l]

                    C2[k, l] += h_[k, l]

                    C3[i, j] += h_[i, j]

                    dS_dC_1[i, j, k, l] = (
                        psi(C1, K, G) - psi(C2, K, G) - psi(C3, K, G) + psi(C, K, G)
                    ) / (h_[i, j] * h_[k, l])

    dS_dC_2 = np.zeros((3, 3, 3, 3))

    h_ = 2.0 * computePerturbationScalingForArray(C) * h

    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    C1 = np.copy(C)
                    C2 = np.copy(C)
                    C3 = np.copy(C)

                    C1[i, j] += h_[i, j]
                    C1[k, l] += h_[k, l]

                    C2[k, l] += h_[k, l]

                    C3[i, j] += h_[i, j]

                    dS_dC_2[i, j, k, l] = (
                        psi(C1, K, G) - psi(C2, K, G) - psi(C3, K, G) + psi(C, K, G)
                    ) / (h_[i, j] * h_[k, l])

    return 2.0 * dS_dC_1 - dS_dC_2


def computeSecondPiolaKirchhoffStress(
    C: np.ndarray, K: float, G: float, h: float = 1e-3
) -> np.ndarray:
    return 2.0 * dPsi_dC(C, K, G, h)


def computeConsistentTangentFromAnalyticalStress(
    C, K, G, h: float = np.finfo(float).eps ** (1.0 / 5)
) -> np.ndarray:
    return dS_dC(C, K, G, h)


def computeConsistentTangentFromPotential(
    C: np.ndarray, K: float, G: float, h: float = np.finfo(float).eps ** (1.0 / 8)
) -> np.ndarray:
    return 2.0 * d2Psi_dC_dC(C, K, G, h)
