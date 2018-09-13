import argparse

import numpy as np
import matplotlib.pyplot as plt


def read_file(filename):
    results = []
    with open(filename) as _file:
        for line in _file:
            try:
                results.append([float(k) for k in line.split('\t')])
            except:
                continue
    results = np.array(results)
    return results


def plot_results(results):
    fig, ax1 = plt.subplots()
    ax1.plot(results[:, 1], results[:, 3], 'o-r', label='Leakage')
    ax1.set_xlabel("Leakage importance")
    ax1.set_ylabel("Leakage/bit")
    ax2 = ax1.twinx()
    ax2.semilogy(results[:, 1], results[:, 2], 'o-g', label="BER")
    ax2.set_ylabel("BER")
    fig.legend()
    fig.tight_layout()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_file")
    args = parser.parse_args()
    results = read_file(args.results_file)
    plot_results(results)

if __name__ == "__main__":
    main()
    plt.show()
