import os
import ast

import numpy as np
import pandas as pd

CONFIG = "config2"

def main():
    for dirpath, dirnames, filenames in os.walk(CONFIG):
        for combination in filenames:
            if not combination.startswith("codewords"):
                continue
            newdir = "{}-{}".format(CONFIG, os.path.splitext(combination)[0])
            print(newdir)
            os.mkdir(newdir)
            codebook_file = os.path.join(dirpath, combination)
            messages, codewords = _parse_codebook_file(codebook_file)
            write_codebook_files(messages, codewords, outdir=newdir)

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

def write_codebook_files(messages, codewords, outdir='.'):
    with open(os.path.join(outdir, "codebook-all.csv"), 'w') as outfile:
        for _message, _codeword in zip(messages, codewords):
            outfile.write("\"{}\",\"{}\"\n".format(list(_message), list(_codeword)))
    idx_rev = np.unique(messages, axis=0, return_inverse=True)[1]
    for num, _mess_idx in enumerate(np.unique(idx_rev)):
        _idx = np.where(idx_rev == _mess_idx)[0]
        _relevant_codewords = codewords[_idx]
        _relevant_message = messages[_idx]
        with open(os.path.join(outdir, "codebook-{}.csv".format(num)), 'w') as outfile:
            for _message, _codeword in zip(_relevant_message, _relevant_codewords):
                outfile.write("\"{}\",\"{}\"\n".format(list(_message), list(_codeword)))

if __name__ == "__main__":
    main()
