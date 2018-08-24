import numpy as np
from digcommpy import messages, decoders, encoders, channels, modulators, metrics
from gauss_mix import entropy_gauss_mix_upper, entropy_gauss_mix_lower

def leakage(code_book_mod, snr_db):
    dim = np.shape(code_book_mod)[1]
    noise_power = 1./(10.**(snr_db/10.))
    sig = noise_power * np.eye(dim)
    entr_gauss_mix = entropy_gauss_mix_upper(code_book_mod, sig)
    #entr_gauss_mix = entropy_gauss_mix_lower(code_book_mod, sig)
    entr_noise = .5*np.log((2*np.pi*np.e)**dim*np.linalg.det(sig))
    return (entr_gauss_mix - entr_noise)/np.log(2)

def main(n=16, k=4, snr_bob=5., snr_eve=0., test_snr=5.):
    channel = "BAWGN"
    encoder = encoders.PolarWiretapEncoder(n, channel, channel,
                                           snr_bob, snr_eve)
    k = encoder.info_length
    print("k={}\tk_bob={}".format(encoder.info_length, encoder.info_length_bob))
    if k > 10:
        return -1, -1, k
    decoder = decoders.PolarWiretapDecoder(n, 'BAWGN', test_snr,
                                           pos_lookup=encoder.pos_lookup)
    modulator = modulators.BpskModulator()
    channel = channels.BawgnChannel(test_snr) #, rate=k/n

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
    results = main()
