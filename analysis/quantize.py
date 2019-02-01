import os.path
import argparse

import numpy as np
from sklearn import cluster
from digcommpy import parsers

def quantize_codebook(codewords, n_clusters=2, modulation='ask'):
    """Quantize a given codebook

    Parameters
    ----------
    codewords : array
        Array of codewords before quantization
    n_clusters : int
        Number of different modulation points, e.g. n_clusters=2 for BPSK
    modulation : str
        Modulation scheme. Possible choices are: 'ask'
    
    Returns
    -------
    codewords : array
        Quantized codewords
    """
    km = cluster.KMeans(n_clusters)
    cluster_pred = km.fit_predict(codewords.reshape(-1, 1))
    choices = km.cluster_centers_.squeeze()
    #quant_cw = np.choose(cluster_pred, choices)
    quant_cw = choices[cluster_pred]
    quant_cw = quant_cw.reshape(codewords.shape)
    return quant_cw

def plot_codebook_hist(codewords, export=False, filename='histogram.dat'):
    codewords = np.ravel(codewords)
    if export:
        hist, bin_edges = np.histogram(codewords, bins=30, density=True)
        hist = np.append(hist, [0])
        with open(filename, 'w') as hist_file:
            hist_file.write("Edge\tHist\n")
            for _edge, _hist in zip(bin_edges, hist):
                hist_file.write("{}\t{}\n".format(_edge, _hist))
    plt.hist(np.ravel(codewords), bins=30)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("codebook")
    parser.add_argument("-c", "--clusters", type=int, default=4)
    parser.add_argument("--wiretap", action='store_true')
    parser.add_argument("--export", action='store_true')
    args = vars(parser.parse_args())
    codebook = parsers.read_codebook_file(args['codebook'], args['wiretap'])[0]
    codewords = np.array(list(codebook.values()))
    plot_codebook_hist(codewords, args['export'],
            os.path.join(os.path.dirname(args['codebook']), 'hist_codebook.dat'))
    quant_codewords = quantize_codebook(codewords, args['clusters'])
    print(np.shape(codewords), np.shape(quant_codewords))
    print(np.unique(np.ravel(quant_codewords), return_counts=True))
    plot_codebook_hist(quant_codewords, args['export'],
            os.path.join(os.path.dirname(args['codebook']),
                         'hist_codebook-c{}.dat'.format(args['clusters'])))
    plt.show()

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    main()
