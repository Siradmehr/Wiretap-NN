import numpy as np
from digcommpy import messages, decoders, encoders, channels, modulators, metrics
from gauss_mix import entropy_gauss_mix_upper, entropy_gauss_mix_lower
from map_decoder import MapDecoder

def leakage(code_book_mod, snr_db):
    dim = np.shape(code_book_mod)[1]
    noise_power = 1./(10.**(snr_db/10.))
    sig = noise_power * np.eye(dim)
    entr_gauss_mix = entropy_gauss_mix_upper(code_book_mod, sig)
    #entr_gauss_mix = entropy_gauss_mix_lower(code_book_mod, sig)
    #entr_noise = .5*np.log((2*np.pi*np.e)**dim*np.linalg.det(sig))
    entr_noise = .5*dim*np.log(2*np.pi*np.e*noise_power)
    return (entr_gauss_mix - entr_noise)/np.log(2)

def main(n=16, k=4, snr_bob=5., snr_eve=0., test_snr=5., alg='ref'):
    channel = "BAWGN"
    encoder = encoders.PolarWiretapEncoder(n, channel, channel,
                                           snr_bob, snr_eve)
    k = encoder.info_length
    k_bob = encoder.info_length_bob
    print("k={}\tk_bob={}".format(k, k_bob))
    if k > 10:
        return -1, -1, k

    modulator = modulators.BpskModulator()
    channel = channels.BawgnChannel(test_snr) #, rate=k/n
    if alg == "ref":
        decoder = decoders.PolarWiretapDecoder(n, 'BAWGN', test_snr,
                                               pos_lookup=encoder.pos_lookup)
    elif alg == "map":
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

    info_book = messages.generate_data(k, binary=True)
    code_book = encoder.encode_messages(info_book)
    code_book_mod = modulator.modulate_symbols(code_book)
    leak = leakage(code_book_mod, snr_eve)

    test_set = messages.generate_data(k, number=10000, binary=True)
    test_code = encoder.encode_messages(test_set)
    test_mod = modulator.modulate_symbols(test_code)
    rec_mod = channel.transmit_data(test_mod)
    pred_info = decoder.decode_messages(rec_mod, channel)
    ber = metrics.ber(test_set, pred_info)
    print("BER:\t{}\nLeak:\t{}".format(ber, leak))
    return ber, leak, k


if __name__ == "__main__":
    snr_bob = -20.
    snr_eve = -50.
    results = main(n=512, snr_bob=snr_bob, snr_eve=snr_eve, test_snr=snr_bob, alg='ref')
