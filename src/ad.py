import numpy as np
from num_dual import Dual64 as D64
from num_dual import HyperDual64 as HD64

from .compressibleNeoHooke import psi
from .compressibleNeoHooke import dPsi_dC as dPsi_dC_analytical


def hyperDual33FromDouble33(C):

    hdC_list = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

    for i in range(3):
        for j in range(3):
            hdC_list[i][j] = D64(C[i, j], 0.0)

    return np.array(hdC_list, dtype=D64)


def hyperDualSecond33FromDouble33(C):

    hdC_list = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

    for i in range(3):
        for j in range(3):
            hdC_list[i][j] = HD64(C[i, j], 0.0, 0.0, 0.0)

    return np.array(hdC_list, dtype=HD64)


def dPsi_dC(C: np.ndarray, K: float, G: float, h: float = 1.0) -> np.ndarray:

    hdC = hyperDual33FromDouble33(C)

    dPsi_dC_ = np.zeros_like(C)

    for i in range(3):
        for j in range(3):
            hdC_right = np.copy(hdC)
            hdC_right[i, j] = D64(C[i, j], h)
            dPsi_dC_[i, j] = psi(hdC_right, K, G).first_derivative / h

    return dPsi_dC_


def dPsi_dC_dC(C: np.ndarray, K: float, G: float, h: float = 1.0) -> np.ndarray:

    hdC = hyperDualSecond33FromDouble33(C)

    dPsi_dC_dC_ = np.zeros((3, 3, 3, 3))

    for i in range(3):
        for j in range(3):
            hdC_right = np.copy(hdC)
            hdC_right[i, j] = HD64(C[i, j], h, 0.0, 0.0)
            for k in range(3):
                for l in range(3):
                    hdC_right[k, l] += HD64(0.0, 0.0, h, 0.0)
                    dPsi_dC_dC_[i, j, k, l] = psi(hdC_right, K, G).second_derivative / (
                        h * h
                    )
                    hdC_right[k, l] -= HD64(0.0, 0.0, h, 0.0)

    return dPsi_dC_dC_


def dS_dC(C: np.ndarray, K: float, G: float, h: float = 1.0) -> np.ndarray:

    hdC = hyperDual33FromDouble33(C)

    dS_dC_ = np.zeros((3, 3, 3, 3))

    for k in range(3):
        for l in range(3):
            hdC_right = np.copy(hdC)
            hdC_right[k, l] = D64(C[k, l], h)

            aux = 2.0 * dPsi_dC_analytical(hdC_right, K, G)
            for i in range(3):
                for j in range(3):
                    dS_dC_[i, j, k, l] = aux[i, j].first_derivative / h

    return dS_dC_


def computeSecondPiolaKirchhoffStress(
    C: np.ndarray, K: float, G: float, h: float = 1.0
) -> np.ndarray:
    return 2.0 * dPsi_dC(C, K, G, h=h)


def computeConsistentTangentFromAnalyticalStress(C, K, G, h: float = 1.0) -> np.ndarray:
    return dS_dC(C, K, G, h=h)


def computeConsistentTangentFromPotential(C, K, G, h: float = 1.0) -> np.ndarray:
    return 2.0 * dPsi_dC_dC(C, K, G, h=h)
