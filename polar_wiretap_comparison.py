# Autoencoders for flexible wiretap code design in Python
#
# Copyright (C) 2019 Karl-Ludwig Besser
# License: GPL Version 3

import os.path

import numpy as np
from digcommpy import messages, decoders, encoders, channels, modulators, metrics
from digcommpy import information_theory as it

from autoencoder_wiretap import calc_wiretap_leakage_ub


def main(n=16, k=4, snr_bob=5., snr_eve=0., test_snr=5., alg='ref'):
    channel = "BAWGN"
    encoder = encoders.PolarWiretapEncoder(n, channel, channel,
                                           snr_bob, snr_eve,
                                           info_length_bob=7, random_length=3)
    k = encoder.info_length
    k_bob = encoder.info_length_bob
    print("k={}\tk_bob={}".format(k, k_bob))
    modulator = modulators.BpskModulator()
    channel = channels.BawgnChannel(test_snr, rate=k_bob/n, input_power=1.)
    if alg == "ref":
        decoder = decoders.PolarWiretapDecoder(n, 'BAWGN', snr_bob,
                                               pos_lookup=encoder.pos_lookup)
    else:
        raise NotImplementedError("Only the standard polar decoder is "
                                  "implemented right now.")
    info_book, code_book, random_book = encoder.generate_codebook(return_random=True)
    code_book_mod = modulator.modulate_symbols(code_book)
    #noise_var_eve = 1./(2*k/n*10.**(snr_eve/10.))
    noise_var_eve = 1./(2*k_bob/n*10.**(snr_eve/10.))
    #print(noise_var_eve)
    write_codebook_files(info_book, code_book_mod, random_book)
    leak = calc_wiretap_leakage_ub(info_book, code_book_mod, noise_var_eve)
    test_set = messages.generate_data(k, number=100000, binary=True)
    test_code = encoder.encode_messages(test_set)
    test_mod = modulator.modulate_symbols(test_code)
    rec_mod = channel.transmit_data(test_mod)
    pred_info = decoder.decode_messages(rec_mod, channel)
    ber = metrics.ber(test_set, pred_info)
    bler = metrics.bler(test_set, pred_info)
    print("BER:\t{}\nBLER:\t{}\nLeak UB:\t{}".format(ber, bler, leak))
    results_file = "Polar_{0}-n{1}-k{2}-r{3}-B{4}E{5}.dat".format(
        alg, n, k, k_bob-k, snr_bob, snr_eve)
    results_file = os.path.join("results", results_file)
    with open(results_file, 'w') as outf:
        outf.write("BER\tBLER\tLeak\n")
        outf.write("{}\t{}\t{}\n".format(ber, bler, leak))
    return ber, leak, k

def write_codebook_files(messages, codewords, random):
    results_file = "codewords-polar.dat"
    #results_file = os.path.join(dirname, results_file)
    with open(results_file, 'w') as outf:
        outf.write("message\trandom\tcodeword\n")
        for _info, _rand, _cw in zip(messages, random, codewords):
            outf.write("{}\t{}\t{}\n".format(list(_info), list(_rand), list(_cw)))

if __name__ == "__main__":
    snr_bob = 0.
    snr_eve = -5.
    results = main(n=64, snr_bob=snr_bob, snr_eve=snr_eve, test_snr=snr_bob, alg='ref')
