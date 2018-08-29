import argparse
import itertools
from datetime import datetime

import polar_wiretap_comparison
from plot_grid_results import _write_results_to_file


def _create_grid(limits_bob, limits_eve):
    snr_bob = range(*limits_bob, 2)
    snr_eve = range(*limits_eve, 2)
    print("{} different combinations".format(len(snr_bob)*len(snr_eve)))
    grid = itertools.product(snr_bob, snr_eve)
    return grid

def run_grid_simulation(n, snr_grid):
    alg = "map"
    results = {}
    for snr_bob, snr_eve in snr_grid:
        print("Running: {}\t{}".format(snr_bob, snr_eve))
        if snr_bob <= snr_eve:
            ber, leak, k = (-1, -1, 0)
        else:
            ber, leak, k = polar_wiretap_comparison.main(
                n, snr_bob=snr_bob, snr_eve=snr_eve, test_snr=snr_bob, alg=alg)
        results[(snr_bob, snr_eve)] = [ber, leak, k]
    return results

def main_grid():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", type=int, nargs="+", dest="limits_eve", default=[-10, 5])
    parser.add_argument("-b", type=int, nargs="+", dest="limits_bob", default=[-5, 10])
    parser.add_argument("-n", type=int, default=16)
    args = vars(parser.parse_args())
    snr_grid = _create_grid(args['limits_bob'], args['limits_eve'])
    results = run_grid_simulation(args['n'], snr_grid)
    _write_results_to_file(results, name="Polar", n=args['n'])



if __name__ == "__main__":
    main_grid()
