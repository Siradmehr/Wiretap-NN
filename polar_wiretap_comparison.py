import os.path

import numpy as np
from digcommpy import messages, decoders, encoders, channels, modulators, metrics
from digcommpy import information_theory as it
#from gauss_mix import entropy_gauss_mix_upper, entropy_gauss_mix_lower

from autoencoder_wiretap import calc_wiretap_leakage


def main(n=16, k=4, snr_bob=5., snr_eve=0., test_snr=5., alg='ref'):
    #snr_eb_n0_bob = snr_bob*(k/n) #Es/N0 = Eb/(N0*R) --> Eb/N0 = R*Es/N0
    #snr_eb_n0_eve = snr_eve*(k/n)
    channel = "BAWGN"
    encoder = encoders.PolarWiretapEncoder(n, channel, channel,
                                           snr_bob, snr_eve)
    k = encoder.info_length
    k_bob = encoder.info_length_bob
    print("k={}\tk_bob={}".format(k, k_bob))
    if k > 10:
        return -1, -1, k

    modulator = modulators.BpskModulator()
    #channel = channels.BawgnChannel(test_snr, rate=k/n, input_power=1.)
    channel = channels.BawgnChannel(test_snr, rate=k_bob/n, input_power=1.)
    #channel = channels.BawgnChannel(test_snr, input_power=1.)
    if alg == "ref":
        decoder = decoders.PolarWiretapDecoder(n, 'BAWGN', snr_bob,
                                               pos_lookup=encoder.pos_lookup)
    elif alg == "map":
        raise NotImplementedError("The MAP decoder is not implemented yet.")
        decoder = MapDecoder(n, k)
        _all_cw = messages.generate_data(k_bob, binary=True)
        train_info = _all_cw[:, encoder.pos_lookup[encoder.pos_lookup < 0] == -1]
        train_code = encoders.PolarEncoder(n, k_bob, "BAWGN", test_snr).encode_messages(_all_cw)
        #train_info = messages.generate_data(k, number=2**k_bob*100, binary=True)
        #train_code = encoder.encode_messages(train_info)
        train_code = modulator.modulate_symbols(train_code)
        #train_code = channel.transmit_data(train_code)
        decoder.train_system((train_code, train_info))
        print(decoder.decoder.p_j_Ci_x_dict)

    #info_book = messages.generate_data(k, binary=True)
    #code_book = encoder.encode_messages(info_book)
    info_book, code_book = encoder.generate_codebook()
    code_book_mod = modulator.modulate_symbols(code_book)
    noise_var_eve = 1./(2*k_bob/n*10.**(snr_eve/10.))
    #noise_var_eve = 1./(2*k/n*10.**(snr_eve/10.))
    #print(noise_var_eve)
    leak = calc_wiretap_leakage(info_book, code_book_mod, noise_var_eve)

    #test_set = messages.generate_data(k, number=100000, binary=True)
    test_set = messages.generate_data(k, number=100, binary=True)
    test_code = encoder.encode_messages(test_set)
    test_mod = modulator.modulate_symbols(test_code)
    rec_mod = channel.transmit_data(test_mod)
#    rec_mod = test_mod + np.sqrt(10.**(-test_snr/10.))*np.random.randn(*np.shape(test_mod))
    pred_info = decoder.decode_messages(rec_mod, channel)
    ber = metrics.ber(test_set, pred_info)
    bler = metrics.bler(test_set, pred_info)
    print("BER:\t{}\nBLER:\t{}\nLeak:\t{}".format(ber, bler, leak))
    results_file = "Polar_{0}-n{1}-k{2}-r{3}-B{4}E{5}.dat".format(
        alg, n, k, k_bob-k, snr_bob, snr_eve)
    results_file = os.path.join("results", results_file)
    with open(results_file, 'w') as outf:
        outf.write("BER\tBLER\tLeak\n")
        outf.write("{}\t{}\t{}\n".format(ber, bler, leak))
    return ber, leak, k


if __name__ == "__main__":
    snr_bob = 0.
    snr_eve = -5.
    results = main(n=16, snr_bob=snr_bob, snr_eve=snr_eve, test_snr=snr_bob, alg='ref')
