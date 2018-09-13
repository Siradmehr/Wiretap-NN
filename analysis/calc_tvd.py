import argparse
import time
import ast

import numpy as np
from scipy import integrate
from scipy import optimize
from scipy.stats import multivariate_normal

from digcommpy import information_theory as it


def single_tvd(all_means, mess_means, sigma):
    dim = np.shape(all_means)[1]
    p_z = it.GaussianMixtureRv(all_means, sigma)
    p_zm = it.GaussianMixtureRv(mess_means, sigma)
    def difference(x):
        _diff = p_z.pdf(x) - p_zm.pdf(x)
        out = [0]*len(x)
        out[0] = _diff
        return out
    roots = []
    for mu in p_z.mu:
        _init_guesses_pos = mu + p_z.sigma[0][0]*np.eye(len(mu))
        _init_guesses_neg = mu - p_z.sigma[0][0]*np.eye(len(mu))
        for x0 in np.vstack((_init_guesses_pos, _init_guesses_neg)):
            intersect = optimize.root(difference, x0)
            #print("Found: {}. Diff: {}".format(intersect.x, difference(intersect.x)))
            roots.append(intersect.x)
    roots = np.array(roots)
    roots = np.unique(roots, axis=0)
    intervals_pos = {}
    intervals_neg = {}
    _eps = np.finfo(float).eps
    for _dim in range(dim):
        for root in roots:
            _root = root[_dim]
            root[_dim] = _root - _eps
            _left = difference(root)
            root[_dim] = _root + _eps
            _right = difference(root)
            print(_left, _right)

    return np.array(roots)


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
    cw = [[-1, 1, 1], [0, 0, 1], [-1, 0, 1], [1, 1, 1], [-1, -2, 3]]
    mm = [[-1, 1, 1], [0, 0, 1]]
    sigma = np.eye(3)
    time_start = time.time()
    integral = single_tvd(cw, mm, sigma)
    print(integral)
    time_end = time.time()
    print("It took {:.1f} sec.".format(time_end-time_start))

if __name__ == "__main__":
    main()
