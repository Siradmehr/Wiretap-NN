import argparse
import time
import ast
from itertools import product

import numpy as np
from scipy import integrate
from scipy import optimize
from scipy.stats import multivariate_normal

from digcommpy import information_theory as it


def product_gauss_mix(all_mu, mess_mu, sigma):
    _sigma = 2*sigma**2
    weights = []
    mu = []
    for mu1, mu2 in product(all_mu, mess_mu):
        weights.append(multivariate_normal.pdf(mu1, mean=mu2, cov=_sigma))
        mu.append((mu1+mu2)/2.)
    mu = np.array(mu)
    weights = np.array(weights)
    return weights, mu, _sigma


def import_codebook(filename):
    results = [[], [], []]
    with open(filename) as infile:
        for line in infile:
            if line.startswith("mess"):
                continue
            parts = [ast.literal_eval(k) for k in line.split("\t")]
            for _part, _result in zip(parts, results):
                _result.append(_part)
    results = [np.array(k) for k in results]
    return results

def main():
    #parser = argparse.ArgumentParser()
    #parser.add_argument("filename")
    #args = parser.parse_args()
    #messages, random, codewords = import_codebook(args.filename)
    cw = np.array([[-1, 1, 1], [0, 0, 1], [-1, 0, 1], [1, 1, 1], [-1, -2, 3]])
    mm = np.array([[-1, 1, 1], [0, 0, 1]])
    sigma = 1.
    time_start = time.time()
    weights_product, mu_product, sigma_product = product_gauss_mix(cw, mm, sigma)
    print(mu_product)
    time_end = time.time()
    print("It took {:.1f} sec.".format(time_end-time_start))

if __name__ == "__main__":
    main()
