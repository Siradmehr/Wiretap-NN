import numpy as np
from digcommpy import messages, decoders, encoders, channels, modulators
from gauss_mix import entropy_gauss_mix_upper

def leakage(code_book_mod):
    sig = eve_noise * np.eye(np.shape(code_book_mod[1]))
    entr_gauss_mix = entropy_gauss_mix_upper(code_book_mod, sig)
    entr_gauss_mix = K.repeat_elements(entr_gauss_mix, batch_size, 0)
    entr_noise = .5*np.log((2*np.pi*np.e)**dim*np.linalg.det(sigma))
    return K.mean(entr_gauss_mix - entr_noise, axis=-1)

def main(n=16, k=4, train_snr={'bob': 5., 'eve': 0.}, test_snr=5.):
    channel = "BAWGN"
    encoder = encoders.PolarWiretapEncoder(n, channel, channel,
                                           train_snr['bob'], train_snr['eve'])
    k = encoder.info_length
    info_book = messages.generate_data(k, binary=True)
    code_book = encoder.encode_messages(info_book)
    leakage = 
    test_set = messages.generate_data(k, number=10000, binary=True)
    test_code = encoder.encode_messages(test_set)

    modulator = modulators.BpskModulator()
    test_mod = modulator.modulate_symbols(test_code)
    channel = channels.BawgnChannel(test_snr) #, rate=k/n


if __name__ == "__main__":
    main()





def main(n=16, k=4, train_snr={'bob': 0., 'eve': 0.}, test_snr=5.):
    #target_eve = .5*np.ones(np.shape(info_book))
    target_eve = np.zeros((len(info_book), n))
    #target_eve = np.zeros((n,))
    loss_weights = [[0., 1.], [.1, .9], [.2, .8], [.3, .7], [.4, .6], [.5, .5],
                    [.6, .4], [.7, .3], [.8, .2], [.9, .1], [1., 0.]]
    results_file = 'loss_weight_combinations-B{bob}E{eve}.dat'.format(**train_snr)
    with open(results_file, 'w') as outf:
        outf.write("wB\twE\tBER\tLeak\tLoss\n")
    for combination in loss_weights:
        print("Loss weight combination: {}".format(combination))
        model = create_model(n, k, symmetrical=True, loss_weights=combination,
                             train_snr=train_snr)
        #print("Start training...")
        history = model.fit([info_book], [info_book, target_eve], epochs=40000,
                            verbose=0, batch_size=2**k)
        #history = model.fit([info_book], [info_book], epochs=400, verbose=0, batch_size=2**k)
        #return history, model
        #print("Start testing...")
        idx_noise_layer = [type(t) == AlwaysOnGaussianNoise for t in model.layers].index(True)
        test_noise = 1./10**(test_snr/10.)
        model.layers[idx_noise_layer].stddev = np.sqrt(test_noise)
        pred = model.predict(test_set)[0]
        pred_bit = np.round(pred)
        ber = hamming_loss(np.ravel(test_set), np.ravel(pred_bit))
        leak = history.history['codewords_loss'][-1]
        total_loss = history.history['loss'][-1]
        print("BER:\t{}".format(ber))
        print("Leakage:\t{}".format(leak))
        print("Loss:\t{}\n".format(total_loss))
        with open(results_file, 'a') as outf:
            outf.write("{}\t{}\t{}\t{}\t{}\n".format(
                *combination, ber, leak, total_loss))
    plot_history(history)
    return history, model

if __name__ == "__main__":
    history, model = main()
    plt.show()
