import numpy as np
from .compressibleNeoHooke import psi
from .compressibleNeoHooke import dPsi_dC as dPsi_dC_analytical
from .tools import computePerturbationScalingForArray
import math


def dPsi_dC(C: np.ndarray, K: float, G: float, h: float = 1e-20) -> np.ndarray:

    cC = C + 0j * np.zeros_like(C)

    dPsi_dC_ = np.zeros_like(C)

    for i in range(3):
        for j in range(3):
            cC_right = np.copy(cC)
            cC_right[i, j] += 1j * h
            dPsi_dC_[i, j] = (psi(cC_right, K, G).imag) / h

    return dPsi_dC_


def dS_dC(C: np.ndarray, K: float, G: float, h: float = 1e-20) -> np.ndarray:

    cC = C + 0j * np.zeros_like(C)

    dS_dC_ = np.zeros((3, 3, 3, 3))

    for k in range(3):
        for l in range(3):
            cC_right = np.copy(cC)
            cC_right[k, l] += 1j * h
            dS_dC_[:, :, k, l] = (2.0 * dPsi_dC_analytical(cC_right, K, G).imag) / h

    return dS_dC_


def d2Psi_dC_dC(
    C: np.ndarray, K: float, G: float, h: float = (np.finfo(float).eps) ** (1.0 / 3)
) -> np.ndarray:

    # Eq. 10 from Ridout (2009)

    dPsi_dC_dC_ = np.zeros((3, 3, 3, 3))

    cC = C + 0j * np.zeros_like(C)

    h_ = computePerturbationScalingForArray(C) * h

    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):

                    cC1 = np.copy(cC)
                    cC1[i, j] += 1j * h_[i, j]
                    cC1[k, l] += h_[k, l]

                    cC2 = np.copy(cC)
                    cC2[i, j] += 1j * h_[i, j]
                    cC2[k, l] -= h_[k, l]

                    dPsi_dC_dC_[i, j, k, l] = (psi(cC1, K, G) - psi(cC2, K, G)).imag / (
                        2.0 * h_[i, j] * h_[k, l]
                    )

    return dPsi_dC_dC_


def computeSecondPiolaKirchhoffStress(
    C: np.ndarray, K: float, G: float, h: float = 1e-20
) -> np.ndarray:
    return 2.0 * dPsi_dC(C, K, G, h=h)


def computeConsistentTangentFromAnalyticalStress(
    C, K, G, h: float = 1e-20
) -> np.ndarray:
    return dS_dC(C, K, G, h=h)


def computeConsistentTangentFromPotential(
    C, K, G, h: float = (np.finfo(float).eps) ** (1.0 / 3)
) -> np.ndarray:
    return 2.0 * d2Psi_dC_dC(C, K, G, h=h)
