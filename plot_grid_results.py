import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors


def _write_results_to_file(results, name, n):
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    filename = "{}-{}-n{}.dat".format(timestamp, name, n)
    with open(filename, 'w') as outfile:
        outfile.write("Bob\tEve\tBER\tLeak\tk\n")
        for snr_combination, values in results.items():
            outfile.write("{}\t{}\t{}\t{}\t{}\n".format(*snr_combination,
                                                        *values))

def plot_results(results):
    results = results[results['BER'] >= 0]
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    sc1 = ax1.scatter(results['Bob'], results['Eve'], c=results['BER'],
                      s=240, norm=colors.LogNorm())
    ax1.set_xlabel("SNR Bob")
    ax1.set_ylabel("SNR Eve")
    ax1.set_title("BER")
    plt.colorbar(sc1, ax=ax1)
    sc2 = ax2.scatter(results["Bob"], results["Eve"],
                      c=results["Leak"]/results["k"], s=240)
    ax2.set_xlabel("SNR Bob")
    ax2.set_ylabel("SNR Eve")
    ax2.set_title("Normalized Leakage")
    plt.colorbar(sc2, ax=ax2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    args = vars(parser.parse_args())
    results = pd.read_csv(args['filename'], sep='\t')
    plot_results(results)
    plt.show()


if __name__ == "__main__":
    main()
