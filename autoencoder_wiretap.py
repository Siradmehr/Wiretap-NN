import os.path

#import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.metrics import hamming_loss
from keras import models
from keras import layers
from keras import backend as K

from digcommpy import messages, metrics
from digcommpy import information_theory as it

DIRNAME = "n{n}k{k}r{r}-B{bob}E{eve}T{train}"

class AlwaysOnGaussianNoise(layers.GaussianNoise):
    def call(self, inputs, training=None):
        return inputs + K.random_normal(shape=K.shape(inputs),
                                        mean=0.,
                                        stddev=self.stddev)

class BinaryNoise(layers.Layer):
    def __init__(self, **kwargs):
        super(BinaryNoise, self).__init__(**kwargs)

    def call(self, inputs, training=None):
        return K.in_train_phase(K.round(K.random_uniform(shape=K.shape(inputs))),
                                inputs, training=training)

class LeakageLayer(layers.Layer):
    def __init__(self, batch_size, noise_power, dim, k, r, **kwargs):
        super(LeakageLayer, self).__init__(**kwargs)
        self.batch_size = batch_size
        self.sigma = noise_pow * np.eye(dim)
        self.k = k
        self.r = r

    def call(self, inputs, training=None):
        def calc_leakage():
            info, codewords = inputs
            info_codewords = codewords #TODO
            h_z = tensor_entropy_gauss_mix_upper(codewords, self.sigma, self.batch_size)
            h_zm = tensor_entropy_gauss_mix_upper(info_codewords, self.sigma, self.batch_size)
        return K.in_train_phase(calc_leakage, inputs, training=training)

def _hard_sigmoid(x):
    #x = (5.*x)-2
    x = (2.5*x)-(2.5*.3)
    #zero = _to_tensor(0., x.dtype.base_dtype)
    #one = _to_tensor(1., x.dtype.base_dtype)
    x = K.clip(x, 0, 1)
    return x

def _fake_not_equal(y_true, y_pred):
    return 1.-K.exp(-20*K.abs(y_true-y_pred))

def loss_ber(y_true, y_pred):
    return K.mean(_fake_not_equal(y_true, _hard_sigmoid(y_pred)))*2.  # to normalize
    #return K.mean(K.not_equal(K.round(y_pred), y_true))*2.  # to normalize


def loss_gauss_mix_entropy(y_true, y_pred, batch_size, dim, noise_pow=.5, k=1):
    #return K.variable(np.array([[1]]))
    #return K.max(y_true-y_pred)
    sigma = noise_pow * np.eye(dim)
    entr_gauss_mix = tensor_entropy_gauss_mix_upper(y_pred, sigma, batch_size, dim)
    entr_gauss_mix = K.repeat_elements(entr_gauss_mix, batch_size, 0)
    #entr_noise = .5*np.log((2*np.pi*np.e)**dim*np.linalg.det(sigma))
    entr_noise = .5*dim*np.log(2*np.pi*np.e*noise_pow)
    return K.mean(entr_gauss_mix - entr_noise, axis=-1)/(np.log(2)*k)

def tensor_entropy_gauss_mix_upper(mu, sig, batch_size, dim=None):
    """Calculate the upper bound of the gaussian mixture entropy using the 
    KL-divergence as distance (Kolchinsky et al, 2017)
    """
    #weights = np.ones(batch_size, dtype=float)/batch_size
    weight = 1./batch_size
    #x = K.variable(value=mu)
    x = K.repeat_elements(mu, batch_size, axis=0)
    #x = K.repeat(x, batch_size)
    #mu = K.variable(value=mu)
    mu = K.tile(mu, (batch_size, 1))
    #mu = K.reshape(mu, (batch_size, batch_size, -1))
    #norm = tensor_norm_pdf(x, mu, sig)
    #norm = K.reshape(norm, (batch_size, batch_size, -1))
    #inner_sums = K.sum(weight*norm, axis=1, keepdims=True)
    #log_inner = K.log(inner_sums)
    dim = np.shape(sig)[0]
    #_factor = 1./(np.sqrt((2*np.pi)**dim*np.linalg.det(sig)))
    _factor_log = -(dim/2)*np.log(2*np.pi*sig[0, 0])
    norm_exp = tensor_norm_pdf_exponent(x, mu, sig)
    norm_exp = K.reshape(norm_exp, (batch_size, batch_size, -1))
    #log_inner = np.log(weight*_factor) + K.logsumexp(norm_exp, axis=1, keepdims=True)
    log_inner = np.log(weight) + _factor_log + K.logsumexp(norm_exp, axis=1, keepdims=True)
    outer_sum = K.sum(weight*log_inner, axis=0)
    entropy_kl = dim/2 - outer_sum
    return entropy_kl

def tensor_norm_pdf_exponent(x, mu, sigma):
    _tensor_sig_inv = K.constant(np.linalg.inv(sigma), dtype='float32')
    _exponent = K.dot((x-mu), _tensor_sig_inv)
    _exponent = K.batch_dot(_exponent, (x-mu), axes=1)
    _exponent = -.5*_exponent
    return _exponent

def tensor_norm_pdf(x, mu, sigma):
    dim = np.shape(sigma)[0]
    _factor = 1./(np.sqrt((2*np.pi)**dim*np.linalg.det(sigma)))
    _tensor_sig_inv = K.constant(np.linalg.inv(sigma), dtype='float32')
    _exponent = K.dot((x-mu), _tensor_sig_inv)
    #_exponent = K.dot(_exponent, K.transpose(x-mu))
    #_exponent = K.batch_dot(_exponent, (x-mu), axes=2)
    _exponent = K.batch_dot(_exponent, (x-mu), axes=1)
    _exp = K.exp(-.5*_exponent)
    return _factor*_exp

def create_model(code_length:int =16, info_length: int =4, activation='relu',
                 symmetrical=True, loss_weights=[.5, .5],
                 train_snr={'bob': 0., 'eve': -5.}, random_length=3,
                 nodes_enc=None, nodes_dec=None):
    rate = info_length/code_length
    if nodes_enc is None:
        #nodes_enc = [8*code_length, 4*code_length, code_length]
        nodes_enc = [8*code_length, code_length]
    if nodes_dec is None:
        nodes_dec = [8*code_length]
    train_snr_lin = {k: 10.**(v/10.) for k, v in train_snr.items()}
    input_power = 1.
    train_noise = {k: input_power/(v) for k, v in train_snr_lin.items()}
    noise_layers = {k: AlwaysOnGaussianNoise(np.sqrt(v), input_shape=(code_length,))
                    for k, v in train_noise.items()}
    if symmetrical:
        nodes_dec = reversed(nodes_enc)
    main_input = layers.Input(shape=(info_length,))
    random_input = layers.Input(shape=(random_length,), name='random_input')
    random_bits = BinaryNoise(input_shape=(random_length,))(random_input)
    input_layer = layers.concatenate([main_input, random_bits])
    layer_list_enc = [input_layer]
    for idx, _nodes in enumerate(nodes_enc):
        _new_layer = layers.Dense(_nodes, activation=activation)(layer_list_enc[idx])
        layer_list_enc.append(_new_layer)
    layer_list_enc.append(layers.BatchNormalization(name='codewords')(layer_list_enc[-1]))
    noise_layer_bob = noise_layers['bob'](layer_list_enc[-1])
    #noise_layer_eve = noise_layers['eve'](layer_list_enc[-1])
    layer_list_decoder = [noise_layer_bob]
    for idx, _nodes in enumerate(nodes_dec):
        _new_layer = layers.Dense(_nodes, activation=activation)(layer_list_decoder[idx])
        layer_list_decoder.append(_new_layer)
    output_layer_bob = layers.Dense(info_length, activation='sigmoid',
                                    name='output_bob')(layer_list_decoder[-1])
    #Append decoder layer to Eve
    #output_layer_eve = layers.Dense(info_length, activation='sigmoid')(noise_layer_eve)
    #output_layer_eve = layers.Lambda(
    #    mutual_info_eve, arguments={'noise_pow': train_noise['eve']},
    #    output_shape=(1,), name='output_eve')(noise_layer_eve)
    model = models.Model(inputs=[main_input, random_input],
                         outputs=[output_layer_bob, layer_list_enc[-1]])
    model.compile('adam', loss_weights=loss_weights,#loss_weights=[.8, .2],
                  #loss=['binary_crossentropy', lambda x, y: loss_gauss_mix_entropy(x, y, 2**info_length, code_length, noise_pow=train_noise['eve'], k=info_length)])
                  loss=['mse', lambda x, y: loss_gauss_mix_entropy(x, y, 2**info_length, code_length, noise_pow=train_noise['eve'], k=info_length)])
    #model = models.Model(inputs=[main_input], outputs=[output_layer_bob])
    #model.compile('adam', 'binary_crossentropy')
    return model

def plot_history(history):
    epochs = history.epoch
    hist_loss = history.history
    fig = plt.figure()
    axs = fig.add_subplot(111)
    for loss_name, loss_val in hist_loss.items():
        axs.plot(epochs, loss_val, label=loss_name)
    axs.legend()
    axs.set_xlabel("Training Epoch")
    axs.set_ylabel("Loss Value")
    axs.set_title("N=16, k=4, SNR_Bob=10dB, SNR_Eve=5dB\nweight_Bob=0.5, weight_Eve=0.5")

def _single_simulation(n, k, random_length, loss_weights, train_snr, train_set,
                       target_eve, test_snr, test_set):
    info_train, rnd_train = train_set
    test_info, test_rnd = test_set
    model = create_model(n, k, symmetrical=False, loss_weights=loss_weights,
                            train_snr=train_snr, activation='sigmoid',
                            random_length=random_length)
    #print("Start training...")
    history = model.fit(train_set, [info_train, target_eve],
                        epochs=5000, verbose=0)#, batch_size=2**k)
    #print("Start testing...")
    idx_noise_layer = [type(t) == AlwaysOnGaussianNoise for t in model.layers].index(True)
    test_noise = 1./10**(test_snr/10.)
    model.layers[idx_noise_layer].stddev = np.sqrt(test_noise)
    pred = model.predict(test_set)[0]
    pred_bit = np.round(pred)
    ber = metrics.ber(test_info, pred_bit)
    bler = metrics.bler(test_info, pred_bit)
    leak = history.history['codewords_loss'][-1]
    total_loss = history.history['loss'][-1]
    print("BER:\t{}".format(ber))
    print("BLER:\t{}".format(bler))
    print("Leak:\t{}".format(leak*k))
    print("Loss:\t{}\n".format(total_loss))
    #with open(results_file, 'a') as outf:
    #    outf.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(
    #        *combination, ber, bler, leak, total_loss))
    _save_codebook(model, k, random_length, combination)
    return ber, leak*k

def loss_weight_sweep(n=16, k=4, train_snr={'bob': 2., 'eve': -5.}, test_snr=0.,
                      random_length=3, loss_weights=[.5, .5], test_size=30000):
    dirname = DIRNAME.format(n=n, k=k, r=random_length, bob=test_snr,
                             eve=train_snr['eve'], train=train_snr['bob'])
    os.makedirs(dirname, exist_ok=True)
    info_book = messages.generate_data(k, binary=True)
    info_train = messages.generate_data(k, number=2**(k+random_length),
                                        binary=True)
    info_train = info_book
    rnd_train = np.zeros((2**k, random_length))
    test_info = messages.generate_data(k, number=test_size, binary=True)
    test_rnd = messages.generate_data(random_length, number=test_size,
                                      binary=True)
    test_set = [test_info, test_rnd]
    target_eve = np.zeros((len(info_train), n))
    #target_eve = np.zeros((n,))
    #loss_weights = [[0., 1.], [.1, .9], [.2, .8], [.3, .7], [.4, .6], [.5, .5],
    #                [.6, .4], [.7, .3], [.8, .2], [.9, .1], [1., 0.]]
    #_weights = np.linspace(0.13, .14, 5)  # 5000 epochs
    #_weights = np.linspace(0.1, .6, 10)
    _weights = np.linspace(0.01, .1, 25)
    loss_weights = [[1.-k, k] for k in _weights]
    results_file = 'lwc-B{bob}E{eve}-T{0}-n{1}-k{2}-r{3}.dat'.format(test_snr, n, k, random_length, **train_snr)
    results_file = os.path.join("data", results_file)
    with open(results_file, 'w') as outf:
        outf.write("wB\twE\tBER\tBLER\tLeak\tLoss\n")
    for combination in loss_weights:
        print("Loss weight combination: {}".format(combination))
        model = create_model(n, k, symmetrical=False, loss_weights=combination,
                             train_snr=train_snr, activation='sigmoid',
                             random_length=random_length)
        #print("Start training...")
        history = model.fit([info_train, rnd_train], [info_train, target_eve],
                            epochs=7000, verbose=0)#, batch_size=2**k)
        #history = model.fit([info_book], [info_book], epochs=400, verbose=0, batch_size=2**k)
        #return history, model
        #print("Start testing...")
        idx_noise_layer = [type(t) == AlwaysOnGaussianNoise for t in model.layers].index(True)
        test_noise = 1./10**(test_snr/10.)
        model.layers[idx_noise_layer].stddev = np.sqrt(test_noise)
        pred = model.predict(test_set)[0]
        pred_bit = np.round(pred)
        #ber = hamming_loss(np.ravel(test_set), np.ravel(pred_bit))
        ber = metrics.ber(test_info, pred_bit)
        bler = metrics.bler(test_info, pred_bit)
        leak = history.history['codewords_loss'][-1]
        total_loss = history.history['loss'][-1]
        print("BER:\t{}".format(ber))
        print("BLER:\t{}".format(bler))
        print("Leak:\t{}".format(leak*k))
        print("Loss:\t{}".format(total_loss))
        codebook = _save_codebook(model, k, random_length, combination, dirname)
        leak = calc_wiretap_leakage(*codebook, train_snr['eve'])
        print("Real leak:\t{}\n".format(leak))
        with open(results_file, 'a') as outf:
            outf.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(
                *combination, ber, bler, leak, total_loss))
    #plot_history(history)
    return history, model

def _save_codebook(model, info_length, random_length, combination, dirname='.'):
    _all_cw = messages.generate_data(info_length+random_length, binary=True)
    _all_cw = np.array(_all_cw)
    info, rand = _all_cw[:, :info_length], _all_cw[:, info_length:]
    encoder_out_layer = model.get_layer("codewords")
    layer_output_func = K.function([*model.inputs, K.learning_phase()],
                                   [encoder_out_layer.output])
    codewords = layer_output_func([info, rand, 0])[0]
    results_file = "codewords-{}.dat".format(combination)
    results_file = os.path.join('codewords', dirname, results_file)
    #results_file = os.path.join('codewords', results_file)
    with open(results_file, 'w') as outf:
        outf.write("mess\trand\tcodeword\n")
        for _info, _rand, _cw in zip(info, rand, codewords):
            outf.write("{}\t{}\t{}\n".format(list(_info), list(_rand), list(_cw)))
    return info, rand, codewords

def calc_wiretap_leakage(info, rand, codewords, snr_eve):
    snr_eve_lin = 10.**(snr_eve/10.)
    input_power = 1.
    noise_var = input_power/snr_eve_lin
    entr_z = it.entropy_gauss_mix_upper(codewords, noise_var)
    messages, idx_rev = np.unique(info, axis=0, return_inverse=True)
    entr_zm = []
    for _mess_idx in np.unique(idx_rev):
        _idx = np.where(idx_rev == _mess_idx)[0]
        _relevant_codewords = codewords[_idx]
        entr_zm.append(it.entropy_gauss_mix_lower(_relevant_codewords, noise_var))
    entr_zm = np.mean(entr_zm)
    leak = entr_z - entr_zm
    return leak

if __name__ == "__main__":
    train_snr = {'bob': 2., 'eve': -5}
    history, model = main(n=16, k=4, train_snr=train_snr, test_snr=0.,
                          random_length=3, test_size=100000)
    plt.show()
