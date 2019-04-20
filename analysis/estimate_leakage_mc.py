import os
import argparse

import numpy as np
from joblib import Parallel, cpu_count, delayed
from digcommpy import information_theory as it
from digcommpy import parsers

def estimate_differential_entropy(codebook, noise_var, samples):
    gauss_mix = it.GaussianMixtureRv(list(codebook.values()), noise_var)
    #print("No.Components: {}, Dim: {}".format(len(gauss_mix), gauss_mix.dim()))
    #x_min = np.min(gauss_mix.mu, axis=0) - 5*np.sqrt(noise_var)
    #x_max = np.max(gauss_mix.mu, axis=0) + 5*np.sqrt(noise_var)
    #_samples = (x_max-x_min)*np.random.rand(samples, gauss_mix.dim()) + x_min
    _samples = gauss_mix.rvs(samples)
    entropy = gauss_mix.logpdf(_samples)
    entropy = -np.mean(entropy)
    return entropy

def estimate_leakage_from_codebook(codebook, noise_var, samples):
    codebook, code_info = codebook
    h_z = estimate_differential_entropy(codebook, noise_var, samples)
    h_zm = []
    info_length = code_info['info_length']
    random_length = code_info['random_length']
    for message in range(2**info_length):
        _relevant_messages = [message*2**random_length + k for k in range(2**random_length)]
        _relevant_codewords = {k: codebook[k] for k in _relevant_messages}
        h_zm.append(estimate_differential_entropy(_relevant_codewords,
                                                  noise_var, samples))
    h_zm = np.mean(h_zm)
    print(h_z, h_zm)
    return np.max((h_z - h_zm)/np.log(2), 0)

def calc_leakage_all_codebooks(codebooks, snr, samples):
    num_cores = cpu_count()
    noise_var = 1./(2*10**(snr/10.))
    print(noise_var)
    for dirpath, dirnames, filenames in os.walk(codebooks):
        with open(os.path.join(dirpath, "leakage-estimate.dat"), 'w') as res_file:
            res_file.write("wB\twE\tLeak\n")
        codebooks = [os.path.join(dirpath, k) for k in filenames if k.startswith("codewords")]
        Parallel(n_jobs=num_cores)(delayed(_calc_leakage_parallel)(
            codebook, noise_var, samples, dirpath) for codebook in codebooks)

def _calc_leakage_parallel(codebook, noise_var, samples, dirpath):
    _codebook = parsers.read_codebook_file(codebook, wiretap=True)
    _leak = estimate_leakage_from_codebook(_codebook, noise_var, samples)
    print(_leak)
    _weights = os.path.basename(codebook).split("[", 1)[1]
    _weights = _weights.split("]", 1)[0].split(',')
    with open(os.path.join(dirpath, "leakage-estimate.dat"), 'a') as res_file:
        res_file.write("{}\t{}\t{}\n".format(float(_weights[0]),
                                            float(_weights[1]), _leak))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("codebooks", help='Directory containing the codebooks')
    parser.add_argument("-s", "--snr", help="SNR as Es/N0", type=float, default=-8.59021942641668)
    parser.add_argument("--samples", help='Number of samples for MC estimation', type=int, default=10000)
    args = vars(parser.parse_args())
    calc_leakage_all_codebooks(**args)

if __name__ == "__main__":
    main()
