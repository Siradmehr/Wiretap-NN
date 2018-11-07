import os.path

import numpy as np
from digcommpy import messages, decoders, encoders, channels, modulators, metrics
from digcommpy import information_theory as it

from autoencoder_wiretap import calc_wiretap_leakage


def main(n=16, k=4, snr_bob=5., snr_eve=0., test_snr=5., alg='ref'):
    channel = "BAWGN"
    encoder = encoders.PolarWiretapEncoder(n, channel, channel,
                                           snr_bob, snr_eve)
    k = encoder.info_length
    k_bob = encoder.info_length_bob
    print("k={}\tk_bob={}".format(k, k_bob))
    modulator = modulators.BpskModulator()
    channel = channels.BawgnChannel(test_snr, rate=k/n, input_power=1.)
    if alg == "ref":
        decoder = decoders.PolarWiretapDecoder(n, 'BAWGN', snr_bob,
                                               pos_lookup=encoder.pos_lookup)
    else:
        raise NotImplementedError("Only the standard polar decoder is "
                                  "implemented right now.")
    info_book, code_book = encoder.generate_codebook()
    code_book_mod = modulator.modulate_symbols(code_book)
    noise_var_eve = 1./(2*k/n*10.**(snr_eve/10.))
    write_codebook_files(info_book, code_book_mod)
    leak = calc_wiretap_leakage(info_book, code_book_mod, noise_var_eve)
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

def write_codebook_files(messages, codewords):
    with open("codebook-all.csv", 'w') as outfile:
        for _message, _codeword in zip(messages, codewords):
            outfile.write("\"{}\",\"{}\"\n".format(list(_message), list(_codeword)))
    idx_rev = np.unique(messages, axis=0, return_inverse=True)[1]
    for num, _mess_idx in enumerate(np.unique(idx_rev)):
        _idx = np.where(idx_rev == _mess_idx)[0]
        _relevant_codewords = codewords[_idx]
        _relevant_message = messages[_idx]
        with open("codebook-{}.csv".format(num), 'w') as outfile:
            for _message, _codeword in zip(_relevant_message, _relevant_codewords):
                outfile.write("\"{}\",\"{}\"\n".format(list(_message), list(_codeword)))

if __name__ == "__main__":
    snr_bob = 0.
    snr_eve = -5.
    results = main(n=16, snr_bob=snr_bob, snr_eve=snr_eve, test_snr=snr_bob, alg='ref')
