import os
import argparse

import pandas as pd
import matplotlib.pyplot as plt

import autoencoder_wiretap
from plot_grid_results import _write_results_to_file


def _extract_information(filename):
    n = int(os.path.splitext(filename.split('-n')[1])[0])
    results = pd.read_csv(filename, sep='\t')
    results = results[results['k'] < 8]
    results = results[results['BER'] >= 0]
    return results, n

def run_grid_simulation(ref_results, n):
    results = {}
    for row in ref_results.itertuples():
        snr_bob, snr_eve = row.Bob, row.Eve
        k = row.k
        print("Bob: {}, Eve: {}\tk={}".format(snr_bob, snr_eve, k))
        #ber, leak = autoencoder_wiretap.single_main(n, k, 2., snr_eve, snr_bob,
        ber, leak = autoencoder_wiretap.single_main(n, k, snr_bob, snr_eve, snr_bob,
            loss_weights=[.65, .35])
        results[(snr_bob, snr_eve)] = [ber, leak, k]
    return results

def main_grid():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    args = vars(parser.parse_args())
    ref_results, n = _extract_information(args['filename'])
    results = run_grid_simulation(ref_results, n)
    _write_results_to_file(results, 'Autoencoder', n)
    return results

if __name__ == "__main__":
    main_grid()
