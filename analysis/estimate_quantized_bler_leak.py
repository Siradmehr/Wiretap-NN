import argparse
import os
import sys

import numpy as np
from keras import models
from keras import layers
from keras import backend as K
from digcommpy import metrics
from digcommpy import messages
from digcommpy import parsers
from digcommpy import encoders
from joblib import Parallel, delayed, cpu_count

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from custom_layers import (TestOnlyGaussianNoise, BinaryNoise, SimpleNormalization)
from autoencoder_wiretap import _save_codebook
from estimate_leakage_mc import estimate_leakage_from_codebook
from quantize import quantize_codebook

def estimate_bler_leakage_codebook(model_path, snr, samples, k=4, 
                                   random_length=3, n_clusters=2,  **kwargs):
    test_set = messages.generate_data(k+random_length, number=samples, binary=True)
    test_info, test_rnd = test_set[:, :k], test_set[:, k:]
    #_params = os.path.basename(model_path).split("-", 1)[1]
    #_params = _params.rsplit("-", 1)
    #_weights = ast.literal_eval(_params[0])
    _params = os.path.basename(model_path).split("-")[0]
    _weights = [float(k) for k in _params.split("_")]
    autoencoder = models.load_model(model_path,
        custom_objects={"TestOnlyGaussianNoise": TestOnlyGaussianNoise,
                        "BinaryNoise": BinaryNoise,
                        "SimpleNormalization": SimpleNormalization})
    autoencoder.load_weights(model_path)
    cb_info, cb_rand, cb_codewords = _save_codebook(autoencoder, k, random_length, _weights, "/tmp")
    code_length = np.shape(cb_codewords)[1]
    quant_codewords = quantize_codebook(cb_codewords, n_clusters)
    print("Point before quant:\t{}\nPoints after quant:\t{}".format(
          len(np.unique(np.ravel(cb_codewords))), len(np.unique(np.ravel(quant_codewords)))))

    DL_input = layers.Input((code_length,))
    autoencoder_decoder = DL_input
    idx_decoder_start_layer = [idx for idx, k in enumerate(autoencoder.layers) if isinstance(k, TestOnlyGaussianNoise)][0]
    for layer in autoencoder.layers[idx_decoder_start_layer:]:
        autoencoder_decoder = layer(autoencoder_decoder)
    autoencoder_decoder = models.Model(inputs=DL_input, outputs=autoencoder_decoder)

    quant_encoder = encoders.CodebookEncoder(code_length, k, (np.hstack((cb_info, cb_rand)), quant_codewords), True)
    test_codewords = quant_encoder.encode_messages(test_set)
    pred = autoencoder_decoder.predict(test_codewords)#[0]
    pred_bit = np.round(pred)[:, :k]
    ber = metrics.ber(test_info, pred_bit)
    bler = metrics.bler(test_info, pred_bit)
    print("BLER:\t{}".format(bler))
    energy_symbol = np.var(cb_codewords)
    noise_var_eve = energy_symbol/(2*10.**(snr/10.))
    codebook_conv = parsers.convert_codebook_to_dict(cb_info, quant_codewords, cb_rand)
    est_leak = estimate_leakage_from_codebook(codebook_conv, noise_var_eve, samples)
    print("{}\t{}\nBLER\tLeakage\n{}\t{}".format(model_path, _weights, bler, est_leak))
    _outfile = "quant-{}_{}-c{}.dat".format(*_weights, n_clusters)
    _outfile = os.path.join(os.path.dirname(model_path), _outfile)
    with open(_outfile, 'w') as outf:
        outf.write("BLER\tLeak\tBER\n{}\t{}\n".format(bler, est_leak, ber))
    return n_clusters, bler, est_leak, ber

def _single_codebook_estimation(model_path, args):
    results = estimate_bler_leakage_codebook(model_path, **args)
    return results

def _single_cluster_estimation(n_clusters, args):
    results = estimate_bler_leakage_codebook(n_clusters=n_clusters, **args)
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help='Autoencoder model file')
    parser.add_argument("-s", "--snr", help="SNR as Es/N0", type=float, default=-8.59021942641668)
    parser.add_argument("-k", type=int, default=4)
    parser.add_argument("-r", "--random_length", type=int, default=3)
    parser.add_argument("-c", "--clusters", type=int, default=(2,), nargs='+')
    parser.add_argument("--samples", help='Number of samples for estimation', type=int, default=200000)
    args = vars(parser.parse_args())
    num_cores = cpu_count()
    results = Parallel(n_jobs=num_cores)(delayed(_single_cluster_estimation)(k, args)
                                         for k in args['clusters'])
    results_file = "quant_results-{}.dat".format(os.path.splitext(os.path.basename(args['model_path']))[0])
    results_file = os.path.join(os.path.dirname(args['model_path']), results_file)
    with open(results_file, 'w') as rfile:
        rfile.write("Cluster\tBLER\tLeak\tBER\n")
        for _results in results:
            rfile.write("{}\t{}\t{}\t{}\n".format(*_results))
        

if __name__ == "__main__":
    main()
