import argparse
import ast

import numpy as np
import pandas as pd
from digcommpy.information_theory import GaussianMixtureRv

def _parse_codebook_file(codebook_file):
    codebook = pd.read_csv(codebook_file, sep='\t')
    _codebook = []
    _messages = []
    for idx, row in codebook.iterrows():
        _codebook.append(ast.literal_eval(row['codeword']))
        _messages.append(ast.literal_eval(row['mess']))
    codebook = np.array(_codebook)
    messages = np.array(_messages)
    return messages, codebook

def _integral_func(x, dist_z, dist_zm):
    lpdf_z = dist_z.logpdf(x)
    lpdf_zm = dist_zm.logpdf(x)
    #_diff = np.exp(lpdf_z) - np.exp(lpdf_zm)
    _diff = np.maximum(lpdf_z, lpdf_zm) + np.log(1.-np.exp(-np.abs(lpdf_z - lpdf_zm)))
    #return np.abs(np.exp(_diff))
    return np.exp(_diff)

def calc_tvd(dist_z, dist_zm, num_samples=1e5):
    num_samples = int(num_samples)
    sigma = np.diag(dist_z.sigma)
    bound_max = np.max(dist_z.mu, axis=0) + 10*sigma
    bound_min = np.min(dist_z.mu, axis=0) - 10*sigma
    bound_distance = bound_max - bound_min
    samples_x = np.random.uniform(size=(num_samples, len(bound_max)))
    samples_x = bound_distance*samples_x + bound_min
    val_func = _integral_func(samples_x, dist_z, dist_zm)
    y_min = min(val_func)
    y_max = max(val_func)
    samples_y = (y_max-y_min)*np.random.uniform(size=(num_samples,)) + y_min
    area = np.prod(bound_distance) * (y_max-y_min)
    counter = _get_sample_area_counter(val_func, samples_y)
    integral = area * counter / num_samples
    return integral

def _get_sample_area_counter(val_func, samples_y):
    counter = 0
    for _val, _y in zip(val_func, samples_y):
        if _val > 0 and _y > 0 and _val >= _y:
            counter += 1
        elif _val < 0 and _y < 0 and _val <= _y:
            counter -= 1
    return counter

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("codebook")
    parser.add_argument("-n", type=int, default=int(1e5), dest='num_samples')
    args = vars(parser.parse_args())
    dist_z, dist_zm = get_distribution(args['codebook'], sigma=0.61)
    #dist_z = GaussianMixtureRv([[1, 2, 3], [-1, -2, 5], [-2, 3, 1]])
    #dist_zm = GaussianMixtureRv([[-1, -2, 5], [-2, 3, 1]])
    tvd = []
    for _dist_zm in dist_zm:
        tvd.append(calc_tvd(dist_z, _dist_zm, args['num_samples']))
    print(tvd)
    tvd = np.mean(tvd)
    print(tvd)
    return tvd

def get_distribution(codebookfile, sigma=1.):
    messages, codewords = _parse_codebook_file(codebookfile)
    dist_z = GaussianMixtureRv(codewords, sigma)
    dist_zm = []
    idx_rev = np.unique(messages, axis=0, return_inverse=True)[1]
    for num, _mess_idx in enumerate(np.unique(idx_rev)):
        _idx = np.where(idx_rev == _mess_idx)[0]
        _relevant_codewords = codewords[_idx]
        dist_zm.append(GaussianMixtureRv(_relevant_codewords, sigma=sigma))
    return dist_z, dist_zm

if __name__ == "__main__":
    main()
