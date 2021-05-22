import pandas as pd
import numpy as np
from pylab import *
import matplotlib.pyplot as plt


class OEE:
    """Estimation of threshold for Ozone Enhancement Episodes [Crisfanelli 2018]"""

    def __init__(self, ozone, freq=1 / 365.25):
        self.oz = ozone
        self.freq = freq

    def get_cutoff(self, plot=True, num_bins=[50, 20], figname=None):
        fil = self.oz.groupby(self.oz.index.dayofyear).transform("mean")
        oz = self.oz.fillna(fil).fillna(method="ffill").fillna(method="bfill")
        self.filled_ozone = oz
        tList = np.arange(len(oz.index))
        yest = self.fitSine(tList, oz.values, self.freq)
        yest = pd.Series(yest, index=oz.index)
        res = oz - yest
        self.sinFit = yest
        self.sinRes = res

        data = res.dropna()

        mean1, sd1, hist_fit1, bin_centres1, hist1 = self.fit_gaussian(
            data, nbins=num_bins[0]
        )

        data1 = data[np.abs(data) >= np.abs(sd1)]

        mean2, sd2, hist_fit2, bin_centres2, hist2 = self.fit_gaussian(
            data1, nbins=num_bins[1]
        )

        pts = self.solve(mean1, mean2, sd1, sd2)
        pt = np.max(pts)

        oee = self.oz[res >= pt]
        noee = self.oz[res < pt]

        if plot:
            fig, axes = plt.subplots(2, 1, figsize=(13, 12))
            ax = axes[0]
            ax.scatter(oee.index, oee, color="r", label="Enhanced Ozone")
            ax.scatter(noee.index, noee, color="k", label="Background Ozone")
            # ax.plot(self.oz, color="k", label="Original Data")
            ax.plot(yest, color="g", label="Sinusoidal Fit")
            ax.plot(res, color="grey", label="Sine Residue", alpha=0.5)
            ax.axhline(0, color="k", ls=":")
            ax.legend(frameon=False)
            ax.set_xlabel("Year")
            ax.set_ylabel("Ozone [ppbv]")

            ax = axes[1]
            ax.scatter(bin_centres1, hist1, label="Test data", color="k")
            ax.plot(bin_centres1, hist_fit1, color="r", lw=4)
            ax.plot(bin_centres2, hist_fit2, color="g", lw=4)
            [ax.axvline(tpt, ls=":", color="k") for tpt in pts]
            ax.set_title("CutOff = %.1f ppbv" % pt, x=0.2, y=0.85)
            ax.set_xlabel("$ \Delta O_3 $ [ppbv]")
            ax.set_ylabel("Count")

            [ax.minorticks_on() for ax in axes]
            plt.subplots_adjust(hspace=0.25)
            if figname:
                plt.savefig(figname, bbox_inches="tight")
            plt.show()
        return pt, oee

    def fit_gaussian(self, data, p0=[1.0, 0.0, 1.0], nbins=50):
        hist, bin_edges = np.histogram(data, density=False, bins=nbins)
        bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
        coeff, _ = curve_fit(self.gaussian_distribution, bin_centres, hist, p0=p0)
        hist_fit = self.gaussian_distribution(bin_centres, *coeff)
        A, mu, sigma = coeff
        return mu, sigma, hist_fit, bin_centres, hist

    @staticmethod
    def fitSine(tList, yList, freq):
        b = matrix(yList).T
        rows = [[sin(freq * 2 * pi * t), cos(freq * 2 * pi * t), 1] for t in tList]
        A = matrix(rows)
        (w, residuals, rank, sing_vals) = lstsq(A, b)
        phi = atan2(w[1, 0], w[0, 0]) * 180 / pi
        amplitude = norm([w[0, 0], w[1, 0]], 2)
        bias = w[2, 0]
        yest = amplitude * sin(tList * freq * 2 * pi + phi * pi / 180.0) + bias
        return yest

    @staticmethod
    def gaussian_distribution(x, *p):
        A, mu, sigma = p
        return A * np.exp(-((x - mu) ** 2) / (2.0 * sigma ** 2))

    @staticmethod
    def solve(m1, m2, std1, std2):
        a = 1 / (2 * std1 ** 2) - 1 / (2 * std2 ** 2)
        b = m2 / (std2 ** 2) - m1 / (std1 ** 2)
        c = m1 ** 2 / (2 * std1 ** 2) - m2 ** 2 / (2 * std2 ** 2) - np.log(std2 / std1)
        return np.roots([a, b, c])
