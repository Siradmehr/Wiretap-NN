from random import choice

import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

def multivariate_gauss_mix(n, mu, sig, weights=None):
    if weights is None:
        weights = np.ones(len(mu), dtype=float)/len(mu)
    x = np.empty((n, np.shape(mu)[1]))
    mu_sig = list(zip(mu, sig))
    for idx in range(n):
        #TODO: Use numpy.random.choice to select index and support weights
        _mu, _sig = choice(mu_sig)
        x[idx] = multivariate_normal.rvs(mean=_mu, cov=_sig)
    return x


def entropy_gauss_mix_lower(mu, sig, weights=None, alpha=.5):
    """Calculate the lower bound of the gaussian mixture entropy using the
    Chernoff alpha-divergence as distance (alpha=.5 for Bhattacharyya distance)
    (Kolchinsky et al, 2017)
    """
    if weights is None:
        weights = np.ones(len(mu), dtype=float)/len(mu)

    dim = np.shape(mu)[1]
    outer_sum = 0
    for idx_i, c_i in enumerate(weights):
        inner_sum = 0
        for idx_j, c_j in enumerate(weights):
            _sig_alpha = 1./(alpha*(1-alpha))*np.array(sig)
            _pdf_j_alpha = multivariate_normal.pdf(mu[idx_i], mean=mu[idx_j],
                                                   cov=_sig_alpha, allow_singular=True)
            inner_sum += c_j*_pdf_j_alpha
        outer_sum += c_i*np.log(inner_sum)

    entropy_alpha = dim/2 + dim/2*np.log(alpha*(1-alpha)) - outer_sum
    return entropy_alpha


def entropy_gauss_mix_upper(mu, sig, weights=None):
    """Calculate the upper bound of the gaussian mixture entropy using the 
    KL-divergence as distance (Kolchinsky et al, 2017)
    """
    if weights is None:
        weights = np.ones(len(mu), dtype=float)/len(mu)

    dim = np.shape(mu)[1]
    outer_sum = 0
    for idx_i, c_i in enumerate(weights):
        inner_sum = 0
        for idx_j, c_j in enumerate(weights):
            _pdf_j = multivariate_normal.pdf(mu[idx_i], mean=mu[idx_j], cov=sig, allow_singular=True)
            inner_sum += c_j*_pdf_j
        outer_sum += c_i*np.log(inner_sum)

    entropy_kl = dim/2 - outer_sum
    return entropy_kl
