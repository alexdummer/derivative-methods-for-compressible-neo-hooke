import numpy as np
from .compressibleNeoHooke import psi
from .compressibleNeoHooke import dPsi_dC as dPsi_dC_analytical
from .tools import computePerturbationScalingForArray


def dPsi_dC(
    C: np.ndarray, K: float, G: float, h: float = np.finfo(float).eps ** (1.0 / 3)
) -> np.ndarray:

    dPsi_dC_ = np.zeros_like(C)

    h_ = computePerturbationScalingForArray(C) * h

    for i in range(3):
        for j in range(3):
            C_right = np.copy(C)
            C_left = np.copy(C)
            C_right[i, j] += h_[i, j]
            C_left[i, j] -= h_[i, j]
            dPsi_dC_[i, j] = (psi(C_right, K, G) - psi(C_left, K, G)) / (2.0 * h_[i, j])

    return dPsi_dC_


def dS_dC(
    C: np.ndarray, K: float, G: float, h: float = np.finfo(float).eps ** (1.0 / 3)
) -> np.ndarray:

    dS_dC_ = np.zeros((3, 3, 3, 3))

    h_ = computePerturbationScalingForArray(C) * h

    for k in range(3):
        for l in range(3):
            # copy Cauchy-Green deformation
            C_right = np.copy(C)
            C_left = np.copy(C)

            # perturbe Cauchy-Green deformation
            C_right[k, l] += h_[k, l]
            C_left[k, l] -= h_[k, l]

            # compute derivataive of S wrt. C_kl
            dS_dC_[:, :, k, l] = (
                2.0 * dPsi_dC_analytical(C_right, K, G)
                - 2.0 * dPsi_dC_analytical(C_left, K, G)
            ) / (2.0 * h_[k, l])

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
                    C4 = np.copy(C)

                    C1[i, j] += h_[i, j]
                    C1[k, l] += h_[k, l]

                    C2[i, j] -= h_[i, j]
                    C2[k, l] += h_[k, l]

                    C3[i, j] += h_[i, j]
                    C3[k, l] -= h_[k, l]

                    C4[i, j] -= h_[i, j]
                    C4[k, l] -= h_[k, l]

                    dS_dC_[i, j, k, l] = (
                        psi(C1, K, G) - psi(C2, K, G) - psi(C3, K, G) + psi(C4, K, G)
                    ) / (4.0 * h_[i, j] * h_[k, l])

    return dS_dC_


def computeSecondPiolaKirchhoffStress(
    C: np.ndarray, K: float, G: float, h: float = np.finfo(float).eps ** (1.0 / 3)
) -> np.ndarray:
    return 2.0 * dPsi_dC(C, K, G, h=h)


def computeConsistentTangentFromAnalyticalStress(
    C: np.ndarray, K: float, G: float, h: float = np.finfo(float).eps ** (1.0 / 3)
) -> np.ndarray:
    return dS_dC(C, K, G, h=h)


def computeConsistentTangentFromPotential(
    C: np.ndarray, K: float, G: float, h: float = np.finfo(float).eps ** (1.0 / 4)
) -> np.ndarray:
    return 2.0 * d2Psi_dC_dC(C, K, G, h=h)
