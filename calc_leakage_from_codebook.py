import os
import argparse
import ast

import numpy as np
import pandas as pd

from autoencoder_wiretap import calc_wiretap_leakage

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

def _calc_leakage_from_codebook_file(codebook_file, snr_eve):
    messages, codewords = _parse_codebook_file(codebook_file)
    noise_var_eve = 1./(2*10**(snr_eve/10.))
    leak = calc_wiretap_leakage(messages, codewords, noise_var_eve)
    return leak

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("codebook_file")
    parser.add_argument("-e", type=float, default=-11)
    args = parser.parse_args()
    snr_eve = args.e

    if os.path.isdir(args.codebook_file):
        leak = {}
        for subdir, dirs, files in os.walk(args.codebook_file):
            for codebook_file in files:
                if codebook_file.startswith("codewords"):
                    print("Working on {}".format(codebook_file))
                    _file = os.path.join(subdir, codebook_file)
                    leak[codebook_file] = _calc_leakage_from_codebook_file(_file, snr_eve)
    else:
        leak  = _calc_leakage_from_codebook_file(args.codebook_file, snr_eve)
    print("SNR={}dB,\tLeak: {}".format(snr_eve, leak))

if __name__ == "__main__":
    main()
