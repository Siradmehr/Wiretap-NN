import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_result_file(filepath, axs=None):
    data = pd.read_csv(filepath, sep='\t')
    if axs is None:
        fig = plt.figure()
        axs = fig.add_subplot(111)
    #axs.loglog(data["BLER"], data["Leak"], '.-')
    axs.semilogx(data["BLER"], data["Leak"], '.-')
    return axs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_file", nargs="+")
    args = parser.parse_args()
    fig = plt.figure()
    axs = fig.add_subplot(111)
    for _file in args.result_file:
        axs = plot_result_file(_file, axs=axs)
    axs.set_xlabel("BLER")
    axs.set_ylabel("Leakage [bit]")
    plt.show()

if __name__ == "__main__":
    main()
