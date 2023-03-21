import numpy as np

import src.fd as fd
import src.cd as cd
import src.csda as csda
import src.ad as ad
import src.efd as efd
import src.ecd as ecd
import case1, case2

# define methods for investigation
methodsForStepSizeStudy = {
    "FDA-F": fd,
    "EFDA-F": efd,
    "FDA-C": cd,
    "EFDA-C": ecd,
    "CSDA": csda,
    "AD": ad,
}

methodsForErrorComparison = {
    "FDA-F": fd,
    "FDA-C": cd,
    "CSDA": csda,
    "AD": ad,
}


# define step sizes
stepsizes = np.logspace(-16, -1, 500)

if __name__ == "__main__":

    # elastic constants
    K = 3500
    G = 1500

    # deformation gradient
    F = np.eye(3)
    F[0, 1] += 1e-2
    F[1, 2] += 2e-2
    F[0, 2] += 3e-2
    F[0, 0] += 5e-2
    F[1, 1] += 6e-2
    F[2, 2] += 7e-2

    # compute Cauchy-Green deformation
    C = np.transpose(F) @ F

    # case 1
    case1.computeComponentWiseErrors(C, K, G, methodsForErrorComparison)
    case1.runStepSizeStudy(C, K, G, methodsForStepSizeStudy, stepsizes)

    # case 2
    case2.computeComponentWiseErrors(C, K, G, methodsForErrorComparison)
    case2.runStepSizeStudy(C, K, G, methodsForStepSizeStudy, stepsizes)
