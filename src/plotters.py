import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
import seaborn as sns

sns.set_theme(context="paper", style="ticks", font="Arial", font_scale=0.8)
colors = sns.color_palette("rocket_r", 12)
cmap = sns.color_palette("rocket_r", as_cmap=True)

sqrt_eps = np.finfo(float).eps ** (1.0 / 2)


def plotStressDifference(dS, name="test"):

    fig, axs = plt.subplots(nrows=len(dS), figsize=(8 / 2.54, 8 / 2.54))

    axid = 0

    #    axs[0].set_title( name )
    for key, value in dS.items():
        ax = axs[axid]
        cax = ax.matshow(
            value,
            norm=SymLogNorm(1e-14, vmin=1e-15, vmax=1e-4),
            # cmap=sns.color_palette("rocket_r", as_cmap=True),
            interpolation="nearest",
        )
        if axid == 0:
            ax.set_xticks(range(6))
            ax.set_xticklabels(
                [
                    r"$S_{11}$",
                    r"$S_{22}$",
                    r"$S_{33}$",
                    r"$S_{12}$",
                    r"$S_{23}$",
                    r"$S_{31}$",
                ]
            )
            ax.tick_params(length=0)
        if axid != 0:
            ax.set_xticks([])
        ax.set_yticks([])
        axid += 1

        ax.text(-2, 0.2, key)

    fig.colorbar(
        cax,
        spacing="uniform",
        ax=axs,
        location="right",
        label=r"$|S_{IJ} - S^{an}_{IJ}| / S^{an}_{IJ}$",
    )
    plt.savefig("{:}.pdf".format(name.replace(" ", "_")), bbox_inches="tight")


def plotTangentDifference(dA, name="test"):

    fig, axs = plt.subplots(ncols=len(dA), figsize=(19 / 2.54, 6 / 2.54))

    axid = 0

    #    axs[0].set_title( name )
    for key, value in dA.items():
        ax = axs[axid]
        cax = ax.matshow(
            value,
            norm=SymLogNorm(1e-16, vmin=1e-15, vmax=1e-2),
            cmap=cmap,  # sns.color_palette("rocket_r", n_colors=9, as_cmap=True),
            interpolation="nearest",
        )

        ax.tick_params(length=0)
        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_xticks([0, 1, 2, 3, 4, 5])
        ax.set_xticklabels(
            ["(11)", "(22)", "(33)", "(12)", "(23)", "(31)"], fontsize="xx-small"
        )
        if axid == 0:
            ax.set_yticks([0, 1, 2, 3, 4, 5])
            ax.set_yticklabels(
                ["(11)", "(22)", "(33)", "(12)", "(23)", "(31)"], fontsize="xx-small"
            )
        axid += 1
        ax.set_title(key, y=-0.3)

    axs[0].text(-3, 3.5, "$\mathbb{C}_{(IJ)(KL)}$", rotation=90)

    plt.tight_layout()

    fig.colorbar(
        cax,
        spacing="uniform",
        ax=axs,
        shrink=0.6,
        ticks=[1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14],
        location="right",
        label=r"relative error $e$",
    )

    plt.savefig("{:}.pdf".format(name.replace(" ", "_")), bbox_inches="tight")


def plotStressAndTangentDifference(dS, dA, name="test"):

    fig, axs = plt.subplots(
        ncols=len(dA), nrows=2, sharex=True, figsize=(19 / 2.54, 7.2 / 2.54)
    )

    colid = 0

    #    axs[0].set_title( name )
    for key, value in dA.items():
        ax = axs[:, colid]
        cax = ax[1].matshow(
            value,
            norm=SymLogNorm(1e-16, vmin=1e-15, vmax=1e-2),
            cmap=cmap,
            interpolation="nearest",
        )

        cax = ax[0].matshow(
            dS[key],
            norm=SymLogNorm(1e-16, vmin=1e-15, vmax=1e-2),
            cmap=cmap,
            interpolation="nearest",
        )
        for i in (0, 1):
            ax[i].tick_params(length=0)
            ax[i].set_yticks([])
        ax[i].set_xticks([0, 1, 2, 3, 4, 5])
        ax[i].set_xticklabels(
            ["(11)", "(22)", "(33)", "(12)", "(23)", "(31)"], fontsize="x-small"
        )
        if colid == 0:
            ax[1].set_yticks([0, 1, 2, 3, 4, 5])
            ax[1].set_yticklabels(
                ["(11)", "(22)", "(33)", "(12)", "(23)", "(31)"], fontsize="x-small"
            )

        colid += 1

        ax[1].set_title(key, y=-0.2)

    axs[0, 0].text(-2, 0.2, "$S_{(IJ)}$", fontsize="large")
    axs[1, 0].text(-2.5, 3.5, "$\mathbb{C}_{(IJ)(KL)}$", rotation=90, fontsize="large")

    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)

    plt.tight_layout()

    fig.colorbar(
        cax,
        spacing="uniform",
        ax=axs,
        shrink=0.75,
        ticks=[1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14],
        # pad = 0.05,
        anchor=(0.0, 0.0),
        # panchor=(0.0, 1.0),
        location="right",
        label=r"relative error $e$",
    )

    plt.savefig("{:}.pdf".format(name.replace(" ", "_")), bbox_inches="tight")


def plotCharacteristicStepsizes(ax):
    for i in [2, 3, 6]:
        h = np.finfo(float).eps ** (1.0 / i)
        ax.plot(
            [h, h],
            [1e-16, 1],
            "k:",
        )
        ax.text(h * 1.2, 1e-12, r"(eps)$^{\frac{1}{" + str(i) + "}}$")


def plotErrorOverPerturbationSize(
    perturbationSizes: np.ndarray,
    errors: dict,
    additionalErrors: dict = None,
    name="test",
):

    fig, axs = plt.subplots(
        figsize=(9 / 2.54, 8 / 2.54)
        if additionalErrors is None
        else (19 / 2.54, 8 / 2.54),
        ncols=int(1 if additionalErrors is None else 2),
        sharex=True,
        sharey=True,
    )  #

    i = 0
    for key, val in errors.items():
        if additionalErrors is None:

            if i == 0:
                plotCharacteristicStepsizes(axs)
            axs.plot(perturbationSizes, val, label=key, color=colors[i])
            axs.grid(b=True)
            axs.set_title("tangent operator $\mathbb{C}_{IJKL}$")
        else:
            # axs[0].plot( [sqrt_eps, sqrt_eps], [-1,1], 'k' )
            # axs[0].plot( [sqrt_eps ** ( 2. / 3 ) , sqrt_eps ** (2./ 3)], [-1,1], 'k' )
            # axs[1].plot( [sqrt_eps, sqrt_eps], [-1,1], 'k' )
            # axs[1].plot( [sqrt_eps ** ( 2. / 3 ) , sqrt_eps ** (2./ 3)], [-1,1], 'k' )

            if i == 0:
                plotCharacteristicStepsizes(axs[0])
                plotCharacteristicStepsizes(axs[1])

            axs[0].plot(perturbationSizes, val, label=key, color=colors[i])
            axs[1].plot(
                perturbationSizes, additionalErrors[key], label=key, color=colors[i]
            )
            axs[0].grid(b=True)
            axs[1].grid(b=True)
            axs[0].set_title("second Piola-Kirchhoff stress $S_{IJ}$")
            axs[1].set_title("tangent operator $\mathbb{C}_{IJKL}$")
            axs[1].set_xlabel(r"relative perturbation size $h_{\rm{r}}$")
        i += 2

    ax = axs if additionalErrors is None else axs[0]
    ax.legend()
    ax.set_xlim(1e-14, 1e-1)
    ax.set_ylim(1e-16, 1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"relative perturbation size $h_{\rm{r}}$")
    ax.set_ylabel(r"maximum relative error $e$")
    plt.tight_layout()
    plt.savefig("{name}.pdf".format(name=name))
