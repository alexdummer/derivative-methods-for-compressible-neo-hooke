import numpy as np
from .compressibleNeoHooke import psi
from .compressibleNeoHooke import dPsi_dC as dPsi_dC_analytical
from .tools import computePerturbationScalingForArray


def dPsi_dC(
    C: np.ndarray, K: float, G: float, h: float = np.sqrt(np.finfo(float).eps)
) -> np.ndarray:

    dPsi_dC_ = np.zeros_like(C)

    psi_ = psi(C, K, G)

    h_ = computePerturbationScalingForArray(C) * h

    for i in range(3):
        for j in range(3):
            C_right = np.copy(C)
            C_right[i, j] += h_[i, j]
            dPsi_dC_[i, j] = (psi(C_right, K, G) - psi_) / h_[i, j]

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
            C_right[k, l] += h_[k, l]
            dS_dC_[:, :, k, l] = (
                2.0 * dPsi_dC_analytical(C_right, K, G) - 2.0 * dPsi_dC_analytical_
            ) / h_[k, l]

    return dS_dC_


def d2Psi_dC_dC(
    C: np.ndarray, K: float, G: float, h: float = np.finfo(float).eps ** (1.0 / 4)
) -> np.ndarray:

    dS_dC_ = np.zeros((3, 3, 3, 3))

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

                    dS_dC_[i, j, k, l] = (
                        psi(C1, K, G) - psi(C2, K, G) - psi(C3, K, G) + psi(C, K, G)
                    ) / (h_[i, j] * h_[k, l])

    return dS_dC_


def computeSecondPiolaKirchhoffStress(
    C: np.ndarray, K: float, G: float, h: float = np.sqrt(np.finfo(float).eps)
) -> np.ndarray:
    return 2.0 * dPsi_dC(C, K, G, h=h)


def computeConsistentTangentFromAnalyticalStress(
    C, K, G, h: float = np.finfo(float).eps ** (1.0 / 2)
) -> np.ndarray:
    return dS_dC(C, K, G, h=h)


def computeConsistentTangentFromPotential(
    C: np.ndarray, K: float, G: float, h: float = np.finfo(float).eps ** (1.0 / 4)
) -> np.ndarray:
    return 2.0 * d2Psi_dC_dC(C, K, G, h=h)
