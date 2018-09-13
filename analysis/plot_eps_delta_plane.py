# -*- coding: utf-8 -*-
import argparse

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt


def tvd_upper_bound(kl_div):
    return np.sqrt(.5*kl_div)

def calc_vi(p, n):
    return (np.log2(np.e)**2)/2. * (p**2 + 2*p*n)/(p+n)**2

def calc_vc(p, n1, n2):
    v1 = calc_vi(p, n1)
    v2 = calc_vi(p, n2)
    _vc = (p*n1)/(p+n1)*((1./n2) + 1./(p+n2))*np.log2(np.e)**2
    return v1 + v2 + _vc

def calc_cs(p, n1, n2):
    cm = .5*np.log2(1+p/n1)
    cw = .5*np.log2(1+p/n2)
    return cm-cw

def delta_upper(eps, n, rate, p, n1, n2):
    v1 = calc_vi(p, n1)
    v2 = calc_vi(p, n2)
    cs = calc_cs(p, n1, n2)
    _inner = cs - rate - np.sqrt(v1/n)*stats.norm.isf(eps)
    _inner = np.sqrt(n/v2) * _inner
    return stats.norm.sf(_inner)

def delta_lower(eps, n, rate, p, n1, n2):
    vc = calc_vc(p, n1, n2)
    cs = calc_cs(p, n1, n2)
    #print(np.sqrt(n/vc))
    _inner = np.sqrt(n/vc) * (cs - rate)
    #print(_inner)
    return stats.norm.sf(_inner) - eps

def plot_theoretical_limits(eps, lower, upper):
    #plt.semilogx(eps, lower, '-r', label="Lower Bound")
    #plt.semilogx(eps, upper, '-b', label="Upper Bound")
    idx_lower = (lower > 0) & (eps > 0)
    idx_upper = (upper > 0) & (eps > 0)
    plt.loglog(eps[idx_lower], lower[idx_lower], '-r', label="Lower Bound")
    plt.loglog(eps[idx_upper], upper[idx_upper], '-b', label="Upper Bound")
    plt.xlabel("Epsilon")
    plt.ylabel("Delta")


def theoretical_limits(n, k, snr_bob=0, snr_eve=-5, **kwargs):
    rate = k/n
    p = 1.
    n1 = p/(10**(snr_bob/10.))
    n2 = p/(10**(snr_eve/10.))
    print("Cs={}".format(calc_cs(p, n1, n2)))
    #eps = 1e-3
    eps = np.logspace(-9, 0, num=50)
    #eps = np.logspace(-2, 0, num=50)
    #print("N={}, k={}, R={}, eps={}".format(n, k, rate, eps))
    print("N={}, k={}, R={}".format(n, k, rate))
    delta_low = delta_lower(eps, n, rate, p, n1, n2)
    delta_high = delta_upper(eps, n, rate, p, n1, n2)
    #print(delta_low, delta_high)
    return eps, delta_low, delta_high


def plot_simulation_data(filename, k, xlabel='BLER', ylabel="Leak", **kwargs):
    if filename is None:
        return
    data = pd.read_csv(filename, sep='\t')
    data.sort_values(by=['wE'], inplace=True)
    data = data[data['BLER'] != 0]
    xdata = data[xlabel]
    ydata = data[ylabel]
    ydata = tvd_upper_bound(ydata*k)
    idx = (xdata > 0) & (ydata > 0)
    plt.loglog(xdata[idx], ydata[idx], '-', label='Autoencoder')
    #plt.semilogx(xdata, ydata, '-')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', dest='filename')
    parser.add_argument("-n", type=int, default=64)
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("-b", type=float, dest="snr_bob", default=0.)
    parser.add_argument("-e", type=float, dest="snr_eve", default=-5.)
    options = vars(parser.parse_args())
    eps, delta_low, delta_high = theoretical_limits(**options)
    plot_theoretical_limits(eps, delta_low, delta_high)
    plot_simulation_data(**options)
    plt.title("N={n}, k={k}, SNR_Bob={snr_bob}, SNR_Eve={snr_eve}".format(**options))
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
