import numpy as np

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


def runComparison(C, K, G, methods, writeLatexFiles=False):

    # compute analytical stress and tangent
    analyticalStress = computeSecondPiolaKirchhoffStress(C, K, G)
    analyticalTangent = computeConsistentTangentFromPotential(C, K, G)

    if writeLatexFiles:
        writeLatexMatrixToFile(stressToVoigt(analyticalStress).T, filename="stress.txt")
        writeLatexMatrixToFile(
            stiffnessToVoigt(analyticalTangent), filename="tangent.txt"
        )

    relativeStressErrors = {}
    relativeTangentErrors = {}

    # loop over approximation methods
    for method, module in methods.items():
        currStress = module.computeSecondPiolaKirchhoffStress(C, K, G)
        relativeStressErrors[method] = stressToVoigt(
            computeRelativeError(analyticalStress, currStress)
        )

        currTangent = module.computeConsistentTangentFromPotential(C, K, G)
        relativeTangentErrors[method] = stiffnessToVoigt(
            computeRelativeError(analyticalTangent, currTangent)
        )

    # visualize relative stress and tangent errors
    plotStressAndTangentDifference(
        relativeStressErrors,
        relativeTangentErrors,
        name="case1-stress-and-tangent-error",
    )


def runStepSizeStudy(C, K, G, methods, stepsizes):

    # compute analytical stress and tangent
    analyticalStress = computeSecondPiolaKirchhoffStress(C, K, G)
    analyticalTangent = computeConsistentTangentFromPotential(C, K, G)

    stressErrors = {}
    tangentErrors = {}

    for method in methods.keys():
        stressErrors[method] = []
        tangentErrors[method] = []

    # loop over step sizes
    for h in stepsizes:
        # loop over approximation methods
        for method, module in methods.items():
            currStress = module.computeSecondPiolaKirchhoffStress(C, K, G, h=h)
            currTangent = module.computeConsistentTangentFromPotential(C, K, G, h=h)

            stressErrors[method].append(
                np.max(np.abs(computeRelativeError(analyticalStress, currStress)))
            )
            tangentErrors[method].append(
                np.max(np.abs(computeRelativeError(analyticalTangent, currTangent)))
            )

    # visualize maximum relative errors of stress and tangent components
    plotErrorOverPerturbationSize(
        stepsizes,
        stressErrors,
        additionalErrors=tangentErrors,
        name="case1-errorOverStepsize",
    )
