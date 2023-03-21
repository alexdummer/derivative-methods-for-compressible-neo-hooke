import numpy as np
import matplotlib.pyplot as plt

from src.compressibleNeoHooke import (
    computeSecondPiolaKirchhoffStress,
    computeConsistentTangentFromPotential,
)
from src.tools import (
    stressToVoigt,
    stiffnessToVoigt,
    computeRelativeError,
    writeLatexMatrixToFile,
)

from src.plotters import (
    plotStressDifference,
    plotTangentDifference,
    plotStressAndTangentDifference,
    plotErrorOverPerturbationSize,
)


def computeComponentWiseErrors(C, K, G, methods):

    relativeTangentErrors = {}

    # analytical tangent
    analyticalTangent = computeConsistentTangentFromPotential(C, K, G)

    # loop over approximation methods
    for method, module in methods.items():
        currTangent = module.computeConsistentTangentFromAnalyticalStress(C, K, G)
        relativeTangentErrors[method] = stiffnessToVoigt(
            computeRelativeError(analyticalTangent, currTangent)
        )

    # visualize relative tangent errors
    plotTangentDifference(
        relativeTangentErrors,
        name="case2-tangent-error",
    )


def runStepSizeStudy(C, K, G, methods, stepsizes):

    # compute analytical stress and tangent
    analyticalTangent = computeConsistentTangentFromPotential(C, K, G)

    tangentErrors = {}

    for method in methods.keys():
        tangentErrors[method] = []

    # loop over step sizes
    for h in stepsizes:
        # loop over approximation methods
        for method, module in methods.items():
            currTangent = module.computeConsistentTangentFromAnalyticalStress(
                C, K, G, h=h
            )
            tangentErrors[method].append(
                np.max(np.abs(computeRelativeError(analyticalTangent, currTangent)))
            )

    # visualize maximum relative errors of stress and tangent components
    plotErrorOverPerturbationSize(
        stepsizes, tangentErrors, name="case2-errorOverStepsize"
    )
