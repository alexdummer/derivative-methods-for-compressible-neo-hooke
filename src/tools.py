import numpy as np

# ordering for voigt notation
ordering = [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (2, 0)]


def stressToVoigt(S):
    s = np.zeros((1, 6))

    for a, (i, j) in enumerate(ordering):
        s[0, a] = S[i, j]
        s[0, a] += S[j, i]
        s[0, a] /= 2.0

    return s


def stiffnessToVoigt(dS_dC):

    voigtStiffness = np.zeros((6, 6))
    for a, (i, j) in enumerate(ordering):
        for b, (k, l) in enumerate(ordering):
            voigtStiffness[a, b] = dS_dC[i, j, k, l]
            voigtStiffness[a, b] += dS_dC[j, i, k, l]
            voigtStiffness[a, b] += dS_dC[j, i, l, k]
            voigtStiffness[a, b] += dS_dC[i, j, l, k]
            voigtStiffness[a, b] /= 4.0

    return voigtStiffness


def computeRelativeError(analytical, approximation):

    analytical_flat = np.ravel(analytical)
    approximation_flat = np.ravel(approximation)

    out_flat = np.zeros_like(approximation_flat)

    for i, a in enumerate(analytical_flat):
        if np.abs(analytical_flat[i]) > 1e-14:
            out_flat[i] = np.abs(
                (analytical_flat[i] - approximation_flat[i]) / analytical_flat[i]
            )
        else:
            out_flat[i] = np.abs((analytical_flat[i] - approximation_flat[i]))

    return np.reshape(out_flat, analytical.shape)


def writeLatexMatrixToFile(mat, filename="text.txt"):

    with open(filename, "w+") as f:
        f.write(r"\begin{bmatrix}")
        f.write("\n")
        for i in range(len(mat[:, 0])):
            line = "\t"
            for j in range(len(mat[i, :])):
                line += r"\num{" + str(mat[i, j]) + "} & "
            f.write(line[:-2])
            f.write("\\\\ \n")
        f.write(r"\end{bmatrix}")


def computePerturbationScalingForArray(array: np.ndarray) -> np.ndarray:
    return np.maximum(np.ones_like(array), np.abs(array))
